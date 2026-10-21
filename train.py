import argparse, copy, json, math, os, time, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import DCDataset, collate
from dcnet import DCNet, dcnet_loss, count_params, HORIZON_STEPS

CFG = dict(L=12, steps=list(HORIZON_STEPS), patch=64, s2_lag_days=20, so2_event_q=0.85, mask_aug=0.1, min_origin_idx=12,
           samples_per_epoch=1000, batch=8, epochs=120, lr=3e-4, wd=0.05, warmup=3, ema=0.999, clip=1.0, patience=15,
           splits=dict(train=["2019-01-01", "2022-12-31"], val=["2023-01-01", "2023-12-31"], test=["2024-01-01", "2024-12-31"]))

def to_dev(b, dev): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
def model_in(b):
    res = dict(fast_v=b["fast_v"], fast_m=b["fast_m"], s2_slow=b["s2_slow"], long_mem=b["long_mem"],
               crop=b["crop"], doy_tgt=b["doy_tgt"], doy_org_sc=b["doy_org_sc"])
    if "weather" in b: res["weather"] = b["weather"]
    return res

@torch.no_grad()
def evaluate(model, dl, dev, peak):
    model.eval(); tot, n = 0.0, 0; cov = np.zeros((7, 3)); cn = np.zeros((7, 3)); sse = np.zeros((7, 3)); sst = np.zeros((7, 3))
    for b in dl:
        b = to_dev(b, dev); H, W = b["crop"][2], b["crop"][3]
        o = model(model_in(b)); crop = lambda t: t[..., :H, :W]
        y, ym = crop(b["y"]), crop(b["ym"]); out = {k: (crop(v) if torch.is_tensor(v) and v.dim() >= 4 else v) for k, v in o.items()}
        tot += dcnet_loss(out, y, ym, crop(b["ev"]), peak_thr=peak).item(); n += 1
        inside = ((y >= out["q10"]) & (y <= out["q90"])).float() * ym
        cov += inside.sum((0, 3, 4)).cpu().numpy(); cn += ym.sum((0, 3, 4)).cpu().numpy()
        e = ((y - out["q50"]) ** 2 * ym).sum((0, 3, 4)).cpu().numpy(); sse += e
        mu = (y * ym).sum((0, 3, 4)) / ym.sum((0, 3, 4)).clamp(min=1); sst += (((y - mu[None, :, :, None, None]) ** 2) * ym).sum((0, 3, 4)).cpu().numpy()
    r2 = np.where(sst > 1e-9, 1 - sse / np.maximum(sst, 1e-9), np.nan)
    return tot / max(n, 1), r2, cov / np.maximum(cn, 1)

def main(a):
    cfg = copy.deepcopy(CFG)
    if a.cfg: cfg.update(json.load(open(a.cfg)))
    if a.batch: cfg["batch"] = a.batch
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    # Hardware Safety Rails (RTX 4060 Laptop GPU Protection)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.vram_cap)
        torch.cuda.empty_cache()
        print(f"[Hardware Guard] VRAM fraction capped at {a.vram_cap*100:.0f}% (~{8.0*a.vram_cap:.1f} GB max).")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DCDataset(a.cache, "train", cfg)
    va = DCDataset(a.cache, "val", cfg, tr.norm, tr.s2_stats, tr.wx_norm)
    tl = DataLoader(tr, cfg["batch"], num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=(a.workers > 0))
    vl = DataLoader(va, 1, num_workers=0, collate_fn=collate)
    model = DCNet(3, (tr.H5, tr.W5), use_B=bool(a.use_B), use_s2=bool(a.use_s2), use_weather=bool(a.use_weather)).to(dev)
    print(f"DC-Net initialized on {dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'}):")
    print(f"  Params: {count_params(model):,}")
    print(f"  Weather Augmented: {bool(a.use_weather)}")
    print(f"  Structural Shift B: {bool(a.use_B)}")
    print(f"  Slow S2 Encoder: {bool(a.use_s2)}")

    ema = copy.deepcopy(model).eval()
    nd = [p for n, p in model.named_parameters() if p.ndim <= 1 or n.startswith("prior.W")]
    dd = [p for n, p in model.named_parameters() if not (p.ndim <= 1 or n.startswith("prior.W"))]
    opt = torch.optim.AdamW([dict(params=dd, weight_decay=cfg["wd"]), dict(params=nd, weight_decay=0.0)], lr=cfg["lr"])
    iters = cfg["samples_per_epoch"] // cfg["batch"]; total = iters * cfg["epochs"]; step = 0
    sched = lambda s: min(1, (s + 1) / (cfg["warmup"] * iters)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))
    amp = dev == "cuda"; scaler = torch.amp.GradScaler(enabled=amp); peak = torch.tensor(tr.peak_thr, device=dev)
    os.makedirs(a.out, exist_ok=True); best, bad = 1e9, 0

    for ep in range(cfg["epochs"]):
        model.train(); run = 0.0
        for b in tl:
            for g in opt.param_groups: g["lr"] = cfg["lr"] * sched(step)
            b = to_dev(b, dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                o = model(model_in(b))
            o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}
            loss = dcnet_loss(o, b["y"], b["ym"], b["ev"], peak_thr=peak)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]); scaler.step(opt); scaler.update(); step += 1; run += loss.item()
            with torch.no_grad():
                d = min(cfg["ema"], (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)

        vloss, r2, cov = evaluate(ema, vl, dev, peak)
        print(f"ep {ep:02d} train {run/iters:.4f} val {vloss:.4f} | R2[h=5d,150d] NO2 {r2[0,0]:.3f}/{r2[-1,0]:.3f} CO {r2[0,1]:.3f}/{r2[-1,1]:.3f} | 80%-cov {cov.mean():.3f}", flush=True)

        if vloss < best:
            best, bad = vloss, 0
            torch.save(dict(model=ema.state_dict(), norm=tr.norm.state(), s2=tr.s2_stats,
                            wx=tr.wx_norm.state() if tr.wx_norm else None, cfg=cfg),
                       os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[Early Stop] Triggered at epoch {ep} (patience={cfg['patience']}). Preserving laptop hardware.")
                break

        if a.max_epochs and ep + 1 >= a.max_epochs:
            print(f"Reached requested max_epochs={a.max_epochs}. Stopping.")
            break

        time.sleep(0.3)

    print(f"Training completed successfully. Best val loss: {best:.4f}. Saved checkpoint to {os.path.join(a.out, 'best.pt')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/dcnet")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vram_cap", type=float, default=0.60)
    ap.add_argument("--use_weather", type=int, default=1)
    ap.add_argument("--use_B", type=int, default=1)
    ap.add_argument("--use_s2", type=int, default=1)
    ap.add_argument("--max_epochs", type=int, default=0)
    main(ap.parse_args())

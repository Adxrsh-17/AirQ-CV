"""train.py - Stage B trainer (Rule.md section 7).

Config C (default) = S2 land use + Stage A prior + S5P history + weather history.
Config B           = same model without S2 and without the prior (--config B) -> measures what land use adds.

Checkpoint selection: mean skill over climatology (SS_clim) across horizons x pollutants on the validation
split, in z-space. Note: with the cache ending in 2024 the validation year is also the only held-out year;
report it as validation, and re-run evaluate.py on 2025 once that data exists (rule B1).
"""
import argparse, copy, json, math, os, time, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import DCDataset, collate, S2_CHANNELS, POLL_NAMES
from dcnet import DCNet, dcnet_loss, count_params, HORIZON_STEPS, HORIZONS_DAYS

SPLITS_STAGE1 = dict(train=["2019-01-01", "2023-12-31"], val=["2024-01-01", "2024-12-31"], test=["2025-01-01", "2025-12-31"])
SPLITS_STAGE2 = dict(train=["2019-01-01", "2024-12-31"], val=["2025-01-01", "2025-12-31"], test=["2026-01-01", "2026-12-31"])

CFG = dict(L=12, steps=list(HORIZON_STEPS), horizons=list(HORIZONS_DAYS), patch=64, s2_lag_days=20, so2_event_q=0.85,
           mask_aug=0.1, min_origin_idx=12, samples_per_epoch=800, batch=8, epochs=40, lr=4e-4, wd=0.05, warmup=2,
           ema=0.999, clip=1.0, patience=10, poll_idx=[0], long_mem_dropout=0.2, holdout_block=8, holdout_frac=0.2,
           holdout_seed=0, clim_ridge=2.0, use_s2=True, use_prior=True, use_weather=True, prior_dir=None, aux_dir=None,
           dim=160, n_tau=4, dropout=0.1, lam_peak=0.5, lam_grad=0.05, lam_blob=0.2, lam_occ=0.2,
           splits=SPLITS_STAGE1)


def to_dev(b, dev): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def build_model(cfg, ds):
    kw = dict(n_poll=ds.n_poll, horizons=tuple(cfg["horizons"]), L=cfg["L"], n_wx=ds.n_wx, s2_ch=S2_CHANNELS,
              s2_ratio=ds.s2_ratio, use_s2=cfg["use_s2"], use_prior=cfg["use_prior"], use_weather=cfg["use_weather"],
              dim=cfg["dim"], n_tau=cfg["n_tau"], p=cfg["dropout"])
    return DCNet(**kw), kw


def so2_pos_of(ds): return ds.poll_idx.index(2) if 2 in ds.poll_idx else None


@torch.no_grad()
def validate(model, dl, dev, so2_pos, cfg, Q, P):
    """Pooled z-space skill per horizon x pollutant on valid target pixels (all / held-out blocks)."""
    model.eval()
    acc = {k: np.zeros((2, Q, P)) for k in ("sse", "sse_c", "sse_p", "sst1", "sst2", "n", "cov")}
    tot, nb = 0.0, 0
    for b in dl:
        b = to_dev(b, dev)
        o = model(b)
        tot += dcnet_loss(o, b, so2_pos, cfg["lam_peak"], cfg["lam_grad"], cfg["lam_blob"], cfg["lam_occ"]).item(); nb += 1
        y, ym = b["y"], b["ym"]
        pers = b["clim_tgt"] + (b["last_a"] * b["has"])[:, None]                 # anomaly persistence (raw persistence is in evaluate.py)
        for r, reg in enumerate((torch.ones_like(b["hold"]), b["hold"])):
            m = ym * reg[:, None]                                                   # (B,Q,P,H,W)
            red = (0, 3, 4)
            acc["sse"][r] += (((y - o["q50"]) ** 2) * m).sum(red).cpu().numpy()
            acc["sse_c"][r] += (((y - b["clim_tgt"]) ** 2) * m).sum(red).cpu().numpy()
            acc["sse_p"][r] += (((y - pers) ** 2) * m).sum(red).cpu().numpy()
            acc["sst1"][r] += (y * m).sum(red).cpu().numpy(); acc["sst2"][r] += (y ** 2 * m).sum(red).cpu().numpy()
            acc["n"][r] += m.sum(red).cpu().numpy()
            acc["cov"][r] += (((y >= o["q10"]) & (y <= o["q90"])).float() * m).sum(red).cpu().numpy()
    n = np.maximum(acc["n"], 1)
    sst = acc["sst2"] - acc["sst1"] ** 2 / n
    with np.errstate(all="ignore"):
        res = dict(loss=tot / max(nb, 1),
                   ss_clim=np.where(acc["n"] > 0, 1 - acc["sse"] / np.maximum(acc["sse_c"], 1e-9), np.nan),
                   ss_pers=np.where(acc["n"] > 0, 1 - acc["sse"] / np.maximum(acc["sse_p"], 1e-9), np.nan),
                   r2=np.where(acc["n"] > 0, 1 - acc["sse"] / np.maximum(sst, 1e-9), np.nan),
                   cov=np.where(acc["n"] > 0, acc["cov"] / n, np.nan), n=acc["n"])
    return res


def main(a):
    cfg = copy.deepcopy(CFG)
    if a.cfg: cfg.update(json.load(open(a.cfg)))
    cfg["splits"] = SPLITS_STAGE2 if a.stage == "2" else SPLITS_STAGE1
    cfg["poll_idx"] = [0, 1, 2] if a.poll == "all" else [POLL_NAMES.index(a.poll)]
    if a.config == "B": cfg["use_s2"], cfg["use_prior"] = False, False
    if a.config == "C_noprior": cfg["use_prior"] = False
    cfg["use_weather"] = bool(a.use_weather)
    for k in ("prior_dir", "aux_dir"):
        if getattr(a, k): cfg[k] = getattr(a, k)
    for k, v in (("batch", a.batch), ("epochs", a.epochs), ("samples_per_epoch", a.samples_per_epoch),
                 ("long_mem_dropout", a.long_mem_dropout), ("holdout_frac", a.holdout_frac), ("lr", a.lr)):
        if v is not None: cfg[k] = v
    cfg["seed"] = a.seed
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count() if dev == "cuda" else 0
    if dev == "cuda":
        for d in range(n_gpu): torch.cuda.set_per_process_memory_fraction(a.vram_cap, device=d)

    t0 = time.time()
    tr = DCDataset(a.cache, "train", cfg)
    va = DCDataset(a.cache, "val", cfg, stats=tr.stats)
    print(f"datasets ready in {time.time() - t0:.0f}s | train origins {len(tr.origins)} | val origins {len(va.origins)} "
          f"| hold-out blocks cover {tr.hold.mean():.2f} of the grid", flush=True)
    wi = lambda i: np.random.seed(torch.initial_seed() % 2 ** 32)   # distinct, reproducible numpy stream per worker
    tl = DataLoader(tr, cfg["batch"], num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0, worker_init_fn=wi, pin_memory=dev == "cuda")
    vl = DataLoader(va, 1, num_workers=min(2, a.workers), collate_fn=collate)

    raw, kw = build_model(cfg, tr)
    raw = raw.to(dev)
    model = torch.nn.DataParallel(raw) if (dev == "cuda" and n_gpu > 1 and not a.no_dp) else raw
    so2_pos = so2_pos_of(tr)
    Q, P = len(cfg["steps"]), tr.n_poll
    pn = [POLL_NAMES[c] for c in tr.poll_idx]
    print(f"DC-Net v2 | config {a.config} | params {count_params(raw):,} | device {dev} x{max(n_gpu, 1)} "
          f"| pollutants {pn} | horizons {cfg['horizons']} d | S2 {cfg['use_s2']} prior {cfg['use_prior']} "
          f"weather {raw.use_weather} | splits {cfg['splits']['train']} / {cfg['splits']['val']}", flush=True)

    ema = copy.deepcopy(raw).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
    nd = [p_ for n_, p_ in raw.named_parameters() if p_.ndim <= 1 or n_.endswith(("alpha_raw", "age_k"))]
    dd = [p_ for n_, p_ in raw.named_parameters() if not (p_.ndim <= 1 or n_.endswith(("alpha_raw", "age_k")))]
    opt = torch.optim.AdamW([dict(params=dd, weight_decay=cfg["wd"]), dict(params=nd, weight_decay=0.0)], lr=cfg["lr"])
    iters = cfg["samples_per_epoch"] // cfg["batch"]; total = iters * cfg["epochs"]; step = 0
    sched = lambda s_: min(1.0, (s_ + 1) / (cfg["warmup"] * iters)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s_ / total)))
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,score," + ",".join(f"ssclim_{p}_{h}d" for p in pn for h in cfg["horizons"]) + "\n")
    best, bad = -1e9, 0

    for ep in range(cfg["epochs"]):
        model.train(); run, t_ep = 0.0, time.time()
        for bi, b in enumerate(tl):
            for g in opt.param_groups: g["lr"] = cfg["lr"] * sched(step)
            b = to_dev(b, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                o = model(b)
            o = {k: v.float() for k, v in o.items()}
            loss = dcnet_loss(o, b, so2_pos, cfg["lam_peak"], cfg["lam_grad"], cfg["lam_blob"], cfg["lam_occ"])
            if not torch.isfinite(loss):
                print(f"  non-finite loss at ep {ep} it {bi}; batch skipped", flush=True); opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg["clip"]); scaler.step(opt); scaler.update()
            step += 1; run += loss.item()
            with torch.no_grad():
                d = min(cfg["ema"], (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), raw.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), raw.buffers()): be.copy_(bm)
            if (bi + 1) % 50 == 0:
                print(f"  ep {ep:02d} [{bi + 1:4d}/{iters}] loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)

        v = validate(ema, vl, dev, so2_pos, cfg, Q, P)
        score = float(np.nanmean(v["ss_clim"][0]))
        fmt = lambda arr: " ".join(f"{x:+.3f}" if np.isfinite(x) else "  nan " for x in arr)
        print(f"ep {ep:02d} ({time.time() - t_ep:.0f}s) train {run / max(iters, 1):.4f} val {v['loss']:.4f} "
              f"| mean SS_clim {score:+.4f} | cov80 {np.nanmean(v['cov'][0]):.3f}", flush=True)
        for p_i, pname in enumerate(pn):
            print(f"     {pname:3s} SS_clim [{fmt(v['ss_clim'][0, :, p_i])}]  SS_anomPers [{fmt(v['ss_pers'][0, :, p_i])}]  "
                  f"R2 [{fmt(v['r2'][0, :, p_i])}]  held-out SS_clim [{fmt(v['ss_clim'][1, :, p_i])}]", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{ep},{run / max(iters, 1):.5f},{v['loss']:.5f},{score:.5f}," +
                    ",".join(f"{v['ss_clim'][0, q, p]:.5f}" for p in range(P) for q in range(Q)) + "\n")

        if score > best:
            best, bad = score, 0
            torch.save(dict(model=ema.state_dict(), model_kwargs=kw, cfg=cfg, stats=tr.stats, epoch=ep, score=score,
                            alpha=torch.sigmoid(ema.alpha_raw).cpu().numpy()), os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"early stop at epoch {ep} (patience {cfg['patience']})"); break
    print(f"done. best mean val SS_clim {best:+.4f} -> {os.path.join(a.out, 'best.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/dcnet")
    ap.add_argument("--cfg", default=None, help="JSON file overriding CFG keys")
    ap.add_argument("--config", default="C", choices=["C", "B", "C_noprior"], help="C = land use on, B = land use off")
    ap.add_argument("--prior_dir", default=None, help="folder with prior_{NO2,CO,SO2}.npy from landuse_baseline.py --prior_only 1")
    ap.add_argument("--aux_dir", default=None, help="writable folder for the annual S2 composite cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", "--max_epochs", dest="epochs", type=int, default=None)
    ap.add_argument("--samples_per_epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--vram_cap", type=float, default=0.90)
    ap.add_argument("--use_weather", type=int, default=1)
    ap.add_argument("--stage", default="1", choices=["1", "2"])
    ap.add_argument("--poll", default="all", choices=["NO2", "CO", "SO2", "all"])
    ap.add_argument("--long_mem_dropout", type=float, default=None)
    ap.add_argument("--holdout_frac", type=float, default=None)
    ap.add_argument("--no_dp", action="store_true", help="disable DataParallel")
    main(ap.parse_args())

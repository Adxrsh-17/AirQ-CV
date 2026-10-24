"""Valid-pixel / coverage audit of a DC-Net cache (run after make_cache.py).

Usage:  python check_cache.py --cache /kaggle/tmp/cache [--min_valid 0.10] [--csv cache_audit.csv]

Reports per month: S2 valid-pixel fraction, S5P valid fraction per pollutant, and
flags months that are (almost) empty. Also checks the cache is fully finite.
Memory-mapped: safe on the 1.2 GB S2 array."""
import argparse, json, os, numpy as np

def main(a):
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    ld = lambda n: np.load(os.path.join(a.cache, n + ".npy"), mmap_mode="r")
    s2, s2ok = ld("s2_month"), ld("s2_month_ok")            # (M,12,H8,W8) fp16, (M,H8,W8)
    s5ok, s5m = ld("s5p_month_ok"), ld("s5p_mask")           # (M,3,H5,W5), (T,3,H5,W5)
    M, T = meta["M"], meta["T"]
    months = [d[:7] for d in meta["month_start"]]
    wins = np.array([d[:4] for d in meta["win_start"]])
    pn = ["NO2", "CO", "SO2"][: s5ok.shape[1]]

    rows = []
    for m in range(M):
        ok = np.asarray(s2ok[m]) > 0
        frac = ok.mean()
        # bands of "valid" pixels should be non-degenerate (all zeros => bad georeferencing / empty export)
        zero_valid = float((np.asarray(s2[m][0])[ok] == 0).mean()) if ok.any() else np.nan
        rows.append([months[m], frac, zero_valid] + [float((np.asarray(s5ok[m, c]) > 0).mean()) for c in range(len(pn))])

    print(f"\n=== S2 / monthly S5P valid-pixel fraction (M={M} months) ===")
    print("month    S2valid  S2band0==0 " + " ".join(f"{p:>6}" for p in pn))
    for r in rows:
        flag = "  <-- LOW" if r[1] < a.min_valid else ""
        print(f"{r[0]}   {r[1]:6.3f}   {r[2]:8.3f}  " + " ".join(f"{v:6.3f}" for v in r[3:]) + flag)

    s2f = np.array([r[1] for r in rows])
    print(f"\nS2 months with valid < {a.min_valid:.0%}: {(s2f < a.min_valid).sum()}/{M}  | "
          f"valid<50%: {(s2f < 0.5).sum()}  | median {np.median(s2f):.3f}  min {s2f.min():.3f}  max {s2f.max():.3f}")
    print("S2 months by valid-fraction bucket:",
          {k: int(((s2f >= lo) & (s2f < hi)).sum()) for k, (lo, hi) in {"<10%": (0, .1), "10-50%": (.1, .5), "50-90%": (.5, .9), ">=90%": (.9, 1.01)}.items()})

    print("\n=== 5-day S5P window coverage per year (mean valid-pixel fraction over windows) ===")
    for y in sorted(set(wins)):
        idx = np.where(wins == y)[0]
        cov = [float(np.mean([(np.asarray(s5m[i, c]) > 0).mean() for i in idx])) for c in range(len(pn))]
        empty = [int(sum((np.asarray(s5m[i, c]) > 0).sum() == 0 for i in idx)) for c in range(len(pn))]
        print(f"{y}: " + "  ".join(f"{p} cov={v:.3f} empty_windows={e}/{len(idx)}" for p, v, e in zip(pn, cov, empty)))

    print("\n=== Finiteness ===")
    bad = sum(int((~np.isfinite(np.asarray(s2[m]))).sum()) for m in range(M))
    print(f"s2_month non-finite values: {bad:,}  ({'OK' if bad == 0 else 'FAIL - rebuild cache'})")
    wx = os.path.join(a.cache, "weather_5d.npy")
    if os.path.exists(wx):
        w = np.load(wx); print(f"weather_5d {w.shape} non-finite: {int((~np.isfinite(w)).sum())}  all-zero windows: {int((np.abs(w).sum(1) == 0).sum())}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["month", "s2_valid", "s2_band0_zero_frac"] + [f"{p}_valid" for p in pn]); wr.writerows(rows)
        print("saved", a.csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True)
    ap.add_argument("--min_valid", type=float, default=0.10); ap.add_argument("--csv", default=None)
    main(ap.parse_args())

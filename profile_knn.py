"""One-off profiler: where does a knngraph epoch actually spend its time?
Loads the real corpus, reports vertex-size distribution, then times kNN-graph
build and a forward+backward for a sample of parts on the chosen device."""
import sys, time
import numpy as np
import torch
import json_dataset as jd
import cp_regressor as cpr
import cp_targets as ct

SRC = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\DE00024082\Desktop\JSON"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda"
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
AMP = len(sys.argv) > 4 and sys.argv[4].lower() in ("amp", "1", "true")
K = 16

t0 = time.time()
parts = [p for p in jd.iter_parts(SRC) if p.n_cps > 0]
print(f"load: {len(parts)} parts with CPs in {time.time()-t0:.1f}s")

nv = np.array([p.n_vertices for p in parts])
print(f"verts: min={nv.min()} median={int(np.median(nv))} mean={int(nv.mean())} "
      f"max={nv.max()}  >|{CAP}|: {(nv>CAP).sum()} parts  >60k: {(nv>60000).sum()}")
print(f"total verts in corpus: {nv.sum():,}  (after {CAP} cap: "
      f"{np.minimum(nv,CAP).sum():,})")

model, meta = cpr.build_regressor("knngraph", {"k": K, "c_width": 128,
                                               "n_layers": 4, "global_feat": True})
model = model.to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
use_amp = AMP and DEV == "cuda"
from torch.amp.grad_scaler import GradScaler   # defining module -> pyright-clean
scaler = GradScaler("cuda", enabled=use_amp)
print(f"device={DEV} cap={CAP} amp={'ON' if use_amp else 'off'}")

# sample: a spread of sizes incl. the biggest
order = np.argsort(nv)
sample = list(order[:: max(1, len(order)//12)]) + [order[-1]]
print(f"\n{'n_verts':>8} {'used':>6} {'graph_ms':>9} {'fwd_ms':>8} {'bwd_ms':>8} {'total_ms':>9}")
build_ms = step_ms = 0.0
for i in sample:
    p = parts[int(i)]
    s = cpr.prepare_sample(p, dedup=True)
    v = s["verts_norm"]; tgt = s["target"]; msk = s["mask"]
    n = len(v)
    if n > CAP:
        heat = tgt[:, ct.HEATMAP]; peaks = np.where(heat >= 1-1e-4)[0]
        rest = np.setdiff1d(np.arange(n), peaks, assume_unique=True)
        extra = np.random.RandomState(0).choice(rest, CAP-len(peaks), replace=False)
        idx = np.sort(np.concatenate([peaks, extra]))
        v, tgt, msk = v[idx], tgt[idx], msk[idx]
    used = len(v)

    tb = time.time(); nbr = cpr._knn_graph(v, K); g_ms = (time.time()-tb)*1e3

    x = torch.tensor(v, dtype=torch.float32, device=DEV)
    nb = nbr.to(DEV)
    t = torch.tensor(tgt, dtype=torch.float32, device=DEV)
    m = torch.tensor(msk, dtype=torch.bool, device=DEV)
    if DEV == "cuda": torch.cuda.synchronize()
    tf = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(x, nb)
        loss, _ = cpr.cp_loss(out, t, m, heat_loss="centernet")
    if DEV == "cuda": torch.cuda.synchronize()
    f_ms = (time.time()-tf)*1e3
    tbw = time.time()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if DEV == "cuda": torch.cuda.synchronize()
    b_ms = (time.time()-tbw)*1e3
    assert torch.isfinite(loss), "non-finite loss under amp"

    build_ms += g_ms; step_ms += f_ms + b_ms
    print(f"{n:>8} {used:>6} {g_ms:>9.1f} {f_ms:>8.1f} {b_ms:>8.1f} {g_ms+f_ms+b_ms:>9.1f}")

ns = len(sample)
print(f"\nper-part avg: graph={build_ms/ns:.0f}ms  step(fwd+bwd)={step_ms/ns:.0f}ms")
est1 = (build_ms+step_ms)/ns * len(parts) / 1000
estN = step_ms/ns * len(parts) / 1000
print(f"EST epoch 1 (build+train): {est1:.0f}s  | later epochs (cached graphs): {estN:.0f}s")
print(f"EST epochs in 60 min: ~{int(max(0,(3600-est1)/max(estN,1e-9))+1)}")

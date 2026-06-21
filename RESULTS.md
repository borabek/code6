# Connection-Point Detection — Implementation & Results

This document is the honest deliverable for the connection-point (CP) detection
work: what was implemented, how it was validated, what the numbers are, and where
the limits are. It accompanies the `wiringrobot-cpd` pipeline.

---

## 1. What was built

An end-to-end pipeline that takes a **raw connector CAD model** and produces
**robot-ready connection points** (3D position + outward approach vector +
confidence), with no manual labelling at inference time:

```
 .stp  ──gmsh──►  triangle mesh  ──►  ABB JSON  ──►  knngraph detector  ──►  connection points
 (CAD)            (step_to_json)      (in memory)     (cp_regressor)         (+ approach vectors)
                                                                                    │
                                                                       viz_preds ──►  .obj scene
                                                                                   (Blender/MeshLab)
```

A single command runs it: `python predict.py model.pt part.stp --out preds.json`
(`predict.py` tessellates the STEP internally via gmsh).

The geometric backbone is **`knngraph`** (EdgeConv on a k-nearest-neighbour
graph): it runs on native Windows + CUDA, needs only `torch + scipy + numpy`
(no native geometry libraries), and has a real geometric receptive field so it
generalises across parts.

---

## 2. The binding constraint: training data

The production training corpus (479 real ABB connector parts with ground-truth
connection points) lives on a company machine and is **not reachable** from the
development machine (the corporate VPN is restricted to managed devices). The
user also cannot hand-label their own `.stp` parts (the true CP locations are
unknown). 

**Consequence:** all training here uses **synthetic** data. This document is
therefore framed honestly — it demonstrates a *working, validated pipeline and
method*, not a model proven on the real connector distribution. Closing that gap
requires real labelled connectors (see §6).

---

## 3. Method: making synthetic data trainable

A geometric detector learns a mapping *local geometry → is this a connection
point?*. The first synthetic generator placed CPs at arbitrary spots on a **flat
plate**, where every vertex has an identical neighbourhood — so no such mapping
exists and the model **collapsed** to a flat heatmap (0 detections). This was
confirmed not to be a code bug: the training machinery overfits a single part
perfectly (`cp_regressor` selftest), and decoding the *ground-truth* targets
recovers ~100 % of CPs (`diag_encoding.py`).

The fix was in the **data**, in two steps:

1. **Learnable** (`make_samples_feat.py`): each CP sits on a distinctive raised
   **boss**, giving the model something to key on. → model fires (100 % synthetic
   F1), but over-generalises ("anything raised is a CP") and false-fires on a
   plain box's corners.
2. **Discriminative** (`make_samples_hard.py`): adds non-CP **decoys** (sharp
   cones, recessed pits, box-like blocks) and zero-CP negatives, so the model
   learns CP is a *specific* feature, not any distinctive geometry.
3. **Generalised** (`make_samples_varied.py`): CPs appear as **several** feature
   types (round boss / tall pin / rectangular pad), broadening what the model
   recognises toward the variety real connectors show.

All four generators are reachable from one CLI: `make_samples.py --mode
flat|feat|hard|varied` (default `varied`/`hard`).

---

## 4. Results (synthetic, held-out)

The decode operating point was fixed honestly on a **held-out** split with
`sweep_decode.py` (never selected on the data it reports).

| Model / data | Held-out F1 | Precision | Recall | Loc err | Notes |
|---|---|---|---|---|---|
| flat plate (`make_samples`) | **0 %** | – | – | – | collapsed (data not learnable) |
| boss (`make_samples_feat`) | 100 %* | high | high | 0.75 mm | fires, but over-generalises |
| **discriminative (`make_samples_hard`)** | **96.3 %** | 95.1 % | 97.5 % | 1.12 mm | operating point thr=0.40, votes=1 |
| varied + socket (`make_samples_varied`) | ~86 % | ~81 % | ~95 % | 1.9 mm | 4 CP feature types incl. recessed socket — harder task, lower but honest F1 |

\* boss F1 is on the same easy single-shape distribution (optimistic).

**Discrimination check (false positives on unfamiliar geometry):** a plain STEP
box (no connector features) went from **10 spurious detections** (boss model) to
**1** (discriminative model) — the decoys taught the model to ignore corners.

**GT-decode ceiling** on every synthetic set is F1 ≈ 1.0, i.e. the encode/decode
and targets are not the bottleneck — only the model fit is.

---

## 5. Results (real connectors — honest, unverified)

Two real connector `.stp` files (~70 mm, ~45 k vertices each) were run through
the pipeline with the locked operating point (thr = 0.40):

| Part | Detected CPs | Confidence | On surface? |
|---|---|---|---|
| `wscaduniverse_3001381` | 2 | 0.48–0.52 | yes (≤0.26 mm) |
| `wscaduniverse_3001475` | 1 | 0.52 | yes |

The detections are **on the mesh surface** and plausible in count, but
**confidence is low (0.3–0.5)** versus the confident synthetic firing — the
expected signature of a synthetic→real distribution gap.

**Visual verification (rendered with `render_scene.py`, height-coloured point
clouds in `3d models/`):** both parts are terminal strips with several recessed
**circular openings** (the true connection points). The boss-only model *missed*
these (1–2 detections, off the openings). Re-training with a **recessed-socket**
feature type (`make_samples_varied --mode varied`) removed that blindness — the
model now fires across the opening regions — but at a usable threshold (thr=0.5,
votes=2 → 5–6 points) its **strongest detections still land on the raised rails /
edges** (false positives), because those resemble the synthetic raised features;
the recessed openings give weaker responses. So coverage improved but precision
on real geometry is still poor.

**Iterating with a targeted decoy.** The over-fires landed on the connectors'
long raised **side rails**. Adding a raised-rail *decoy* to the synthetic data
(`make_samples_hard.add_rail`, used by `make_samples_varied`) and retraining cut
the over-firing sharply (18→4 and 23→8 detections at thr=0.40) and moved the
surviving points **off the rail and into the body, onto/near the circular
openings** — a clear, visible improvement (esp. part 3001381; part 3001475
improved too but still shows some edge detections).

**Conclusion:** each time a real false-positive source is identified and added as
a synthetic **decoy**, the model improves on the real parts — the decoy-iteration
loop genuinely works and narrows the synthetic→real gap. But it is incremental
and still bounded (no ground truth; residual edge hits). Reliable, complete
detection still needs real labelled connectors (§6); the decoy iterations narrow
the gap, they do not close it.

---

## 5b. Breakthrough — extract the connection points straight from the STEP CAD

The synthetic→real gap exists because **tessellation throws away the CAD
semantics**. But the connection openings on these connectors are real CAD
features — `CYLINDRICAL_SURFACE` entities — and the STEP keeps them. `step_openings.py`
loads the STEP with gmsh (so coordinates match the tessellated mesh exactly),
finds the cylindrical faces, keeps the terminal-radius ones and merges them into
holes:

```bash
python step_openings.py part.stp --out cp.json     # centre + insert axis + radius
```

On both real connectors this gives **5 connection points each, landing exactly on
the circular openings** (verified in `3d models/*_step_view*.png`) — precise, with
**no ML, no synthetic data, no manual labelling**. This is the cleanest answer
whenever the STEP is available:

- **STEP available → use it directly.** The connection points are in the CAD.
- **Mesh-only / scanned parts → auto-label.** `step_openings.to_abb_labels()` turns
  the CAD-extracted points into ABB ConnectionPoints — REAL training labels — to
  fine-tune the `knngraph` model so it also works on parts that have no STEP. This
  is the bridge that closes the synthetic→real gap with real data.

Caveats: the radius filter (default 1.3–3.0 mm) is per part-family (use `--list`
to find the terminal-radius cluster); only circular/cylindrical openings are
caught (not rectangular slots); the insert-axis sign is approximate for symmetric
through-holes.

## 5c. First real-data training (the #2 bridge, executed)

**14 real connector `.stp` files** (wscaduniverse) were auto-labelled with
`step_openings --label-corpus` (68 connection points, **no manual labelling**) and
used to train the `knngraph` (`train_1h`, CUDA). Tested on the 2 original
connectors held out (their CAD ground truth = 5 CPs each):

- training: VAL F1 ≈ 31 % (micro 39 %), loc ≈ 3 mm, ang ≈ 60°;
- held-out: **over-detects** (9 and 8 vs 5 GT), roughly on the body / opening
  regions but with false positives (e.g. on a rail).

This proves the **whole real-data ML loop runs end to end** — STEP → auto-label →
train → predict — with real labels obtained for free from the CAD. But **14 parts
is far too few** to train a generalising geometric model, so the result is weak.
The fix is **more data (50–100+ parts), not code**: each batch of ~20–30 more
auto-labelled connectors should lift the numbers materially. For STEP-available
parts the CAD-direct path (§5b) stays exact and is the recommended route; the ML
model matters for mesh-only / scanned parts, and needs the larger corpus to be
good.

**Adding 2 more parts (16 total) + heavy augmentation (×12 rigid clones) already
helped, confirming the levers:** VAL F1 31 % → **44 %** (best micro 53 %),
**precision 29 % → 59 %**, loc 3.0 → **2.2 mm**, and visually the detections moved
to the body/openings with no rail false-positives. So on tiny corpora the two
levers are *more parts* and *more augmentation* — but a deployable model still
needs the 50–100+ part corpus. Training converges in ~25 min on 16 parts;
overnight time does not help (data-limited, not time-limited).

Reproduce:
```bash
python step_openings.py ./more_stp --label-corpus ./real_labeled
python train_1h.py --source ./real_labeled --minutes 45 --split-group none \
    --test-frac 0 --run-name cp_real
python predict.py model_cp_real.pt held_out.stp --device cuda --out preds.json
```

## 6. Honest limitations & next steps

- **Not validated on real connectors.** Accuracy on real `.stp` parts is unknown
  (no labels). The synthetic model may miss recessed/socket-type CPs it never saw.
- **The real fix is real data.** Either restore access to the ABB corpus, label a
  few dozen of the user's own `.stp` parts (mark the visible openings — exact
  coordinates aren't needed), or obtain another annotated connector dataset. Even
  20–50 real labelled parts would move this from "pipeline works" to "works on
  your parts".
- **Inference subsamples** ~45 k-vertex parts to ~7 k (matching training density
  on the 4 GB GPU); fine features may be undersampled.

---

## 7. Reproduce

```bash
# 1. generate learnable+discriminative synthetic data
python make_samples.py ./synthetic_varied 500 --mode varied

# 2. train (native Windows + CUDA) with the known-good config
python train_1h.py --source ./synthetic_varied --minutes 35 --split-group none --run-name cp_knn

# 3. lock the decode operating point on a held-out split
python sweep_decode.py checkpoints/cp_knn_best.ckpt ./synthetic_varied --split-group none --holdout-frac 0.3 --min-precision 0.5

# 4. detect on a real STEP file (one command) and visualise
python predict.py model_cp_knn.pt part.stp --device cuda --out preds.json
python step_to_json.py part.stp --out part.json
python viz_preds.py part.json preds.json --out part_scene.obj   # open in Blender/MeshLab

# verify the whole codebase
python run_selftests.py    # 10 modules, all pass
```

When the real corpus is available, one command trains it correctly (known-good
config + held-out operating point baked in):

```bash
python run_full.py /path/to/real/abb/corpus
```

---

## 8. Key files

| File | Role |
|---|---|
| `step_to_json.py` | STEP/STL/OBJ → ABB JSON; `iter_parts_any()` lets `predict.py` read `.stp` directly |
| `make_samples{,_feat,_hard,_varied}.py` | synthetic generators (flat → learnable → discriminative → varied) |
| `train_cp.py` / `cp_regressor.py` | training loop + 3 backbones (knngraph used here) |
| `sweep_decode.py` | honest, held-out decode operating-point selection |
| `predict.py` | inference CLI (accepts `.stp` directly) |
| `viz_preds.py` | overlay detections on the mesh → `.obj` scene |
| `run_full.py` | one-command real-corpus run (known-good + swept config baked in) |

# Training entry point for connection-point detection on ABB JSON data (Option B).
#
# Ties together:
#   json_dataset  – stream parts from the corpus (1 GB single-array file OR a
#                   directory of per-part files)
#   cp_targets    – encode terminal-block locations -> per-vertex targets;
#                   decode predictions -> discrete connection points
#   cp_regressor  – the per-vertex 7-channel head + combined loss + train loop
#   metrics_cp    – localisation (mm) / angular (deg) / precision / recall
#
# Three backbones:
#   --backbone mlp           : light per-vertex MLP on xyz (only needs torch).
#                              Memorises geometry -> good for overfit smoke-tests,
#                              but has no receptive field so it does NOT generalise.
#   --backbone diffusionnet  : the production model (needs the diffusion_net pkg
#                              + robust_laplacian/potpourri3d -> Linux/WSL).
#   --backbone knngraph      : torch+scipy EdgeConv on a kNN graph. Has a real
#                              geometric receptive field (generalises) but NO
#                              native geometry deps -> runs on native Windows.
#
# Memory / 1 GB scale: parts are streamed; with --max-parts you cap how many are
# materialised. The eigenbasis for the diffusionnet backbone is the expensive
# step and should be cached per part (see diffusionnet.precompute_operators).

import os
import json
import logging
import argparse

import numpy as np

import json_dataset as jd
import cp_targets as ct
import cp_regressor as cpr
import metrics as mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# split a corpus by PartNr (deterministic hash -> stable across runs)
# ---------------------------------------------------------------------------

def _bucket(key, seed):
    """Stable hash of a split key -> a float in [0,1)."""
    import hashlib
    h = int(hashlib.sha256((str(seed) + ":" + str(key)).encode()).hexdigest(), 16)
    return (h % 1000) / 1000.0


def split_part_ids(ids, val_frac=0.2, seed=0, group_keys=None):
    """Deterministic train/val split (hash-based, stable).

    group_keys: optional parallel list of GROUP keys (see json_dataset.build_group_keys).
    When given, the bucket is decided per GROUP, so all parts sharing a key (a part
    family / near-duplicate geometry) land on the SAME side -- this is what stops
    sibling variants leaking across train/val. Default (None) hashes each PartNr,
    the original per-part behaviour.
    """
    keys = group_keys if group_keys is not None else ids
    val, train = set(), []
    for pid, key in zip(ids, keys):
        if _bucket(key, seed) < val_frac:
            val.add(pid)
        else:
            train.append(pid)
    return set(train), val


def three_way_split(ids, val_frac=0.2, test_frac=0.0, seed=0, group_keys=None):
    """Train / val / TEST split from one stable hash space, group-aware.

    The held-out test set is carved from the same hash so it is frozen and never
    overlaps train/val: bucket in [0,val_frac) -> val, [val_frac,val_frac+test_frac)
    -> test, else train. test_frac=0 reproduces split_part_ids (test empty). The
    test split is for a FINAL, unbiased report only -- it must never feed training
    or best-checkpoint/decode selection.
    """
    keys = group_keys if group_keys is not None else ids
    train, val, test = [], set(), set()
    for pid, key in zip(ids, keys):
        b = _bucket(key, seed)
        if b < val_frac:
            val.add(pid)
        elif b < val_frac + test_frac:
            test.add(pid)
        else:
            train.append(pid)
    return set(train), val, test


# ---------------------------------------------------------------------------
# evaluate a trained model (any backbone) on prepared samples
# ---------------------------------------------------------------------------

def _nms_radius(s, clearance_mm=5.0):
    """Decode merge radius = the fixed tool clearance, DECOUPLED from sigma.

    Previously this was max(2*sigma, clearance). sigma is ~2% of the bbox diagonal,
    which on an elongated part is tens of mm, so 2*sigma ballooned the merge radius
    ABOVE the spacing between distinct connection points -- decode then collapsed two
    real CPs into one (a guaranteed recall cap, independent of model quality, and
    worse at inference where sigma is the uncapped bbox value). The votes for a
    single CP already cluster tightly because each voting vertex predicts the offset
    TO that CP, so the merge radius only needs to cover offset error + tool
    clearance: a fixed clearance (tunable via --nms-clearance-mm / sweep_decode) is
    the correct, GT-free, train==inference radius. `s` is kept for signature compat.
    """
    return float(clearance_mm)


def _print_part_metrics(rep):
    """One line of per-part keypoint metrics: accuracy / precision / recall / F1
    plus the TP/FP/FN counts and mean loc/ang error. loc/ang show '-' when the
    part had no matched detection (their mean is NaN)."""
    loc = rep["mean_loc_err_mm"]; ang = rep["mean_ang_err_deg"]
    loc_s = "   -   " if (isinstance(loc, float) and np.isnan(loc)) else "%5.2fmm" % loc
    ang_s = "   -   " if (isinstance(ang, float) and np.isnan(ang)) else "%5.1fdeg" % ang
    print("  %-26s acc=%.2f  prec=%.2f  rec=%.2f  F1=%.2f   "
          "TP=%d FP=%d FN=%d   loc=%s ang=%s"
          % (rep["part_nr"], rep["accuracy"], rep["precision"], rep["recall"],
             rep["f1"], rep["tp"], rep["fp"], rep["fn"], loc_s, ang_s))


def evaluate(samples, parts_meta, infer_arr, dist_thresh_mm=5.0,
             heatmap_thresh=0.3, min_votes=1, nms_clearance_mm=5.0, progress=True,
             print_points=False):
    """Decode predictions for each sample and aggregate keypoint metrics.

    infer_arr(sample, part_meta) -> (N,7) numpy array of per-vertex predictions
    (heatmap already sigmoid'd, direction unit). This abstracts the backbone so
    the same decode+metrics path serves both the MLP and DiffusionNet models.
    min_votes: a decoded cluster needs at least this many voting vertices (#6:
    >1 suppresses lone-vertex false positives -> higher precision).

    progress: log a heartbeat every ~20% of parts (only when >=20 parts) so a
    slow eval -- a single 470k-vertex part is ~30s of CPU inference -- is visibly
    making progress instead of looking frozen.
    """
    import time
    reports = []
    n = len(samples)
    hb = max(1, n // 5)
    t0 = time.time()
    for j, (s, pm) in enumerate(zip(samples, parts_meta), 1):
        arr = infer_arr(s, pm)
        preds = ct.decode_predictions(pm["vertices"], arr,
                                      heatmap_thresh=heatmap_thresh,
                                      nms_radius_mm=_nms_radius(s, nms_clearance_mm),
                                      min_votes=min_votes)
        rep = mcp.keypoint_report(preds, s["gt_points"], s["gt_directions"],
                                  dist_thresh_mm=dist_thresh_mm)
        rep["part_nr"] = pm["part_nr"]
        reports.append(rep)
        if print_points:
            _print_part_metrics(rep)
        if progress and n >= 20 and (j % hb == 0 or j == n):
            logger.info("    eval %d/%d parts (%.0fs)", j, n, time.time() - t0)
    return reports


def aggregate(reports):
    """Per-part means (macro) + pooled detection counts (micro) in one dict.

    macro keys (accuracy/precision/recall/f1/loc/ang) average each part's
    score; micro_* pool TP/FP/FN over all parts first, which weights every
    ground-truth point equally and is what 'overall corpus accuracy' means.
    """
    if not reports:
        return {}
    keys = ["accuracy", "precision", "recall", "f1",
            "mean_loc_err_mm", "mean_ang_err_deg"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in reports if not (isinstance(r[k], float) and np.isnan(r[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["n_parts"] = len(reports)
    agg["total_gt"] = sum(r["n_gt"] for r in reports)
    agg["total_tp"] = sum(r["tp"] for r in reports)
    agg["total_fp"] = sum(r["fp"] for r in reports)
    agg["total_fn"] = sum(r["fn"] for r in reports)
    tp, fp, fn = agg["total_tp"], agg["total_fp"], agg["total_fn"]
    agg["micro_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    agg["micro_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    pr = agg["micro_precision"] + agg["micro_recall"]
    agg["micro_f1"] = (2 * agg["micro_precision"] * agg["micro_recall"] / pr
                       if pr else 0.0)
    agg["micro_accuracy"] = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return agg


def _val_metrics_fn(val_s, val_m, infer_builder, dist_thresh_mm, min_votes,
                    heatmap_thresh=0.3, nms_clearance_mm=5.0):
    """Build metrics_fn(model, meta) for the training bookkeeper: decode and
    score the validation parts, returning the aggregate keypoint metrics
    (accuracy / precision / recall / f1 / loc / ang). None when no val set --
    the loop then skips val evaluation and best-by-F1 tracking.

    heatmap_thresh / nms_clearance_mm are the decode knobs that govern the
    precision/recall trade-off; they must match the values used for the final
    train/val evaluate() so the best-by-F1 checkpoint is selected under the same
    decoding the report uses."""
    if not val_s:
        return None

    def metrics_fn(m, mt):
        return aggregate(evaluate(val_s, val_m, infer_builder(m, mt),
                                  dist_thresh_mm=dist_thresh_mm,
                                  heatmap_thresh=heatmap_thresh,
                                  min_votes=min_votes,
                                  nms_clearance_mm=nms_clearance_mm))
    return metrics_fn


# ---------------------------------------------------------------------------
# main train+eval
# ---------------------------------------------------------------------------

def train_and_eval(source, backbone="mlp", epochs=300, val_frac=0.2,
                   max_parts=None, device="cpu", dist_thresh_mm=5.0, seed=0,
                   op_cache_dir=None, max_gpu_verts=60000,
                   best_path=None, last_path=None, history_path=None,
                   save_every=1, resume_from=None,
                   min_votes=1, heat_loss="centernet",
                   focal_gamma=2.0, lr_decay_every=0, lr_decay_rate=0.5,
                   accum_steps=1, low_memory=False, eval_every=10, patience=0,
                   heatmap_thresh=0.3, nms_clearance_mm=5.0, heat_pos_weight=50.0,
                   augment=0, aug_rotate=True, aug_jitter_frac=0.0, aug_reflect=False,
                   split_group="none", test_frac=0.0, cp_surface_tol_frac=None,
                   neg_frac=0.0, max_neg=None,
                   best_metric="micro_f1", resume_strict=False, snapshot_every=0,
                   w_heat=1.0, w_off=5.0, w_dir=1.0, weight_decay=0.0,
                   lr_schedule="step", warmup_epochs=0, grad_clip=0.0,
                   offset_heat_weight=True, part_weight_mode="none",
                   knn_config=None, print_points=False, amp=True):
    """Load parts, prepare targets, train, and evaluate.

    Returns (train_agg, val_agg, train_reports, val_reports, test_agg, test_reports).

    best_path / last_path : full-state .ckpt files (best-by-val-F1, rolling
    resumable snapshot). resume_from continues a previous run from its saved
    epoch toward `epochs` total -- including on another machine after a git
    pull. history_path collects the per-eval val metrics as JSON.

    split_group : 'none' (split each PartNr independently), 'prefix' (group
    part-family variants by stripped PartNr) or 'geometry' (group near-identical
    meshes) -- grouping stops sibling variants leaking across train/val.
    test_frac : carve a frozen held-out TEST split (never trains, never tunes) for
    one unbiased final number. cp_surface_tol_frac : drop CPs farther than this
    fraction of the bbox diagonal from the mesh (off-surface mislabels).
    neg_frac / max_neg : include zero-CP parts (valid negatives) in the FIT set so
    the model learns to stay silent where there is nothing -- only those on the
    train side are used, never val/test.

    augment : number of augmented clones to add per TRAINING part (0 = off).
    Each clone is a random rigid rotation (+ optional reflection / vertex jitter)
    of the part, with CP points/directions transformed consistently. The clones
    enter the FIT set only; the train/val reports use the original parts, so val
    metrics stay deployment-real. The main generalisation lever for the raw-xyz
    backbones (mlp/knngraph), which are otherwise pose-overfit.
    """
    parts = []
    for part in jd.iter_parts(source):
        parts.append(part)
        if max_parts and len(parts) >= max_parts:
            break
    if not parts:
        raise SystemExit(f"no parts found in {source}")
    logger.info("loaded %d parts", len(parts))

    ids = [p.part_nr for p in parts]
    group_keys = jd.build_group_keys(parts, mode=split_group)
    if split_group != "none":
        n_groups = len(set(group_keys))
        logger.info("split-group=%s: %d parts -> %d groups", split_group,
                    len(parts), n_groups)
        # over-aggressive grouping (e.g. 'prefix' on uniformly-named parts) can
        # collapse the whole corpus into a couple of groups, which the hash split
        # then sends to ONE side -> empty val/test and no best-tracking. Warn early.
        if n_groups < 5 and len(parts) >= 10:
            logger.warning("split-group=%s collapsed %d parts into only %d group(s) "
                           "-- the train/val/test split will likely be degenerate. "
                           "Use --split-group geometry (groups by mesh, naming-"
                           "independent) or none if val/test come out empty.",
                           split_group, len(parts), n_groups)
    train_ids, val_ids, test_ids = three_way_split(
        ids, val_frac=val_frac, test_frac=test_frac, seed=seed, group_keys=group_keys)
    # auto-recover from a degenerate GROUPED split: --split-group prefix on
    # uniformly-named parts (e.g. synthetic part_0001..) can collapse the corpus
    # into ~1 group, which the hash then sends to ONE side -> empty val/test and
    # no best-checkpoint tracking. Fall back to a per-part split so the run stays
    # valid regardless of the --split-group / data combination.
    if split_group != "none" and len(parts) >= 10 and (
            not val_ids or (test_frac > 0 and not test_ids)):
        logger.warning("split-group=%s produced a degenerate split (val=%d, test=%d) "
                       "-- falling back to a per-part split", split_group,
                       len(val_ids), len(test_ids))
        train_ids, val_ids, test_ids = three_way_split(
            ids, val_frac=val_frac, test_frac=test_frac, seed=seed, group_keys=None)
    # guarantee at least one part trains even on a tiny corpus
    if not train_ids:
        train_ids = {ids[0]}
        val_ids.discard(ids[0]); test_ids.discard(ids[0])

    # train_s/train_m are the ORIGINAL train parts (used for the train report);
    # fit_s is what the trainer actually optimises on -- originals plus `augment`
    # augmented clones per part (+ optional negatives). Keeping them separate means
    # augmentation never leaks into the reported metrics and val/test stay pristine.
    train_s, train_m, val_s, val_m, fit_s = [], [], [], [], []
    test_s, test_m = [], []
    train_negs = []                          # zero-CP parts on the train side
    aug_rng = np.random.default_rng(seed + 1)
    n_aug = max(0, int(augment))
    for p in parts:
        in_val = p.part_nr in val_ids
        in_test = p.part_nr in test_ids
        if p.n_cps == 0:
            if not in_val and not in_test:   # negatives only from the train side
                train_negs.append(p)
            continue
        s = cpr.prepare_sample(p, dedup=True, cp_surface_tol_frac=cp_surface_tol_frac)
        meta = {"part_nr": p.part_nr, "vertices": p.vertices}
        if in_val:
            val_s.append(s); val_m.append(meta)
        elif in_test:
            test_s.append(s); test_m.append(meta)
        else:
            train_s.append(s); train_m.append(meta)
            fit_s.append(s)
            for _ in range(n_aug):
                ap = cpr.augment_part(p, aug_rng, rotate=aug_rotate,
                                      jitter_frac=aug_jitter_frac, reflect=aug_reflect)
                # quiet=True: a clone is a rigid copy, so its CP-surface/dedup/sigma
                # warnings are identical to the original's -- printing them per clone
                # just floods the log (and falsely implicates augmentation).
                fit_s.append(cpr.prepare_sample(ap, dedup=True, quiet=True,
                                                cp_surface_tol_frac=cp_surface_tol_frac))

    # negatives: a sampled fraction (capped) of the train-side zero-CP parts.
    n_neg = 0
    if neg_frac and neg_frac > 0.0 and train_negs:
        k = int(round(neg_frac * len(train_negs)))
        if max_neg is not None:
            k = min(k, int(max_neg))
        k = max(0, min(k, len(train_negs)))
        if k:
            pick = np.random.default_rng(seed + 2).choice(len(train_negs), k, replace=False)
            for j in pick:
                fit_s.append(cpr.prepare_sample(train_negs[int(j)], dedup=True,
                                                cp_surface_tol_frac=cp_surface_tol_frac))
            n_neg = k
    logger.info("train parts=%d (+%d augmented +%d negatives = %d fit)  "
                "val parts=%d  test parts=%d  (avail zero-CP train-side=%d)",
                len(train_s), len(fit_s) - len(train_s) - n_neg, n_neg, len(fit_s),
                len(val_s), len(test_s), len(train_negs))
    # an empty val set on a real corpus means no best-by-F1 tracking and no honest
    # number -- almost always a degenerate split (over-aggressive --split-group, or
    # too-small --val-frac). Make it impossible to miss.
    if not val_s and len(parts) >= 10:
        logger.warning("VALIDATION SET IS EMPTY -- best-checkpoint tracking is "
                       "disabled and val/test metrics will be blank. Likely cause: "
                       "--split-group=%s collapsed the corpus. Fix with "
                       "--split-group geometry|none or a larger --val-frac.",
                       split_group)

    # Make the big-part recall ceiling VISIBLE: count how many val/test GT points
    # become unrecoverable because the inference subsample drops their region.
    if backbone == "knngraph" and max_gpu_verts:
        for tag, ss in (("val", val_s), ("test", test_s)):
            n_sub = n_gt = n_risk = 0
            for s in ss:
                if len(s["verts_norm"]) > max_gpu_verts:
                    n_sub += 1
                    g, r = cpr.subsample_coverage(s["verts_norm"], s["gt_points"],
                                                  s["center"], s["scale"], max_gpu_verts)
                    n_gt += g; n_risk += r
            if n_sub:
                logger.info("%s: %d part(s) subsampled (>%d verts); %d/%d GT points "
                            "at risk of becoming unrecoverable FNs (%.1f%%)",
                            tag, n_sub, max_gpu_verts, n_risk, n_gt,
                            100.0 * n_risk / max(1, n_gt))

    # decode operating point bundled into every saved checkpoint, so a raw .ckpt
    # deploys with THIS run's knobs (not DEFAULT_DECODE).
    decode = {"heatmap_thresh": heatmap_thresh, "min_votes": min_votes,
              "nms_clearance_mm": nms_clearance_mm, "dist_thresh_mm": dist_thresh_mm}
    # training fingerprint stored in the checkpoint so a --resume can detect a
    # mismatched continuation (incomparable best_f1 / history).
    train_config = {"val_frac": val_frac, "seed": seed, "split_group": split_group,
                    "test_frac": test_frac, "heat_pos_weight": heat_pos_weight,
                    "heat_loss": heat_loss, "focal_gamma": focal_gamma,
                    "augment": augment, "best_metric": best_metric,
                    "max_gpu_verts": max_gpu_verts, "backbone": backbone,
                    "source": str(source), "w_heat": w_heat, "w_off": w_off,
                    "w_dir": w_dir, "weight_decay": weight_decay,
                    "offset_heat_weight": offset_heat_weight,
                    "part_weight_mode": part_weight_mode}

    if backbone == "mlp":
        import torch

        def _infer_builder_mlp(m, mt):
            def f(s, pm):
                m.eval()
                with torch.no_grad():
                    x = torch.tensor(s["verts_norm"], dtype=torch.float32,
                                     device=device)
                    return cpr.pred_to_array(m(x), offset_scale=s["scale"])
            return f
        _infer_builder = _infer_builder_mlp

        metrics_fn = _val_metrics_fn(val_s, val_m, _infer_builder,
                                     dist_thresh_mm, min_votes,
                                     heatmap_thresh=heatmap_thresh,
                                     nms_clearance_mm=nms_clearance_mm)
        model = cpr.train_cpmlp(
            fit_s, epochs=epochs, device=device,
            log_every=max(1, epochs // 6),
            metrics_fn=metrics_fn, eval_every=eval_every, patience=patience,
            best_path=best_path, last_path=last_path, save_every=save_every,
            history_path=history_path, resume_from=resume_from, seed=seed,
            decode=decode, train_config=train_config, best_metric=best_metric,
            resume_strict=resume_strict, snapshot_every=snapshot_every,
            weight_decay=weight_decay, grad_clip=grad_clip,
            offset_heat_weight=offset_heat_weight, part_weight_mode=part_weight_mode,
            lr_schedule=lr_schedule, warmup_epochs=warmup_epochs)
        infer_arr = _infer_builder(model, None)
    elif backbone == "knngraph":
        # torch-only backbone (no native geometry deps) -> runs on native Windows.
        # cache val kNN graphs across evals (same arrays reused every metrics_fn).
        _knn_graph_cache = {}

        def _infer_builder_knn(m, mt):
            def f(s, pm):
                return cpr.infer_knngraph(m, mt, s["verts_norm"], device=device,
                                          max_gpu_verts=max_gpu_verts,
                                          offset_scale=s["scale"],
                                          graph_cache=_knn_graph_cache)
            return f
        _infer_builder = _infer_builder_knn

        metrics_fn = _val_metrics_fn(val_s, val_m, _infer_builder,
                                     dist_thresh_mm, min_votes,
                                     heatmap_thresh=heatmap_thresh,
                                     nms_clearance_mm=nms_clearance_mm)
        model, meta = cpr.train_knngraph_regressor(
            fit_s, config=knn_config, epochs=epochs, device=device,
            log_every=1, max_gpu_verts=max_gpu_verts,
            metrics_fn=metrics_fn, eval_every=eval_every, patience=patience,
            best_path=best_path, last_path=last_path, save_every=save_every,
            history_path=history_path, resume_from=resume_from, seed=seed,
            heat_pos_weight=heat_pos_weight, w_heat=w_heat, w_off=w_off, w_dir=w_dir,
            heat_loss=heat_loss, focal_gamma=focal_gamma,
            lr_decay_every=lr_decay_every, lr_decay_rate=lr_decay_rate,
            accum_steps=accum_steps, amp=amp,
            decode=decode, train_config=train_config, best_metric=best_metric,
            resume_strict=resume_strict, snapshot_every=snapshot_every,
            weight_decay=weight_decay, lr_schedule=lr_schedule,
            warmup_epochs=warmup_epochs, grad_clip=grad_clip,
            offset_heat_weight=offset_heat_weight, part_weight_mode=part_weight_mode)
        infer_arr = _infer_builder(model, meta)
    else:  # diffusionnet
        def _infer_builder_dn(m, mt):
            def f(s, pm):
                return cpr.infer_diffusionnet(
                    m, mt, pm["vertices"], s["faces"], s["verts_norm"],
                    op_cache_dir=op_cache_dir, device=device,
                    max_gpu_verts=max_gpu_verts, offset_scale=s["scale"])
            return f
        _infer_builder = _infer_builder_dn

        metrics_fn = _val_metrics_fn(val_s, val_m, _infer_builder,
                                     dist_thresh_mm, min_votes,
                                     heatmap_thresh=heatmap_thresh,
                                     nms_clearance_mm=nms_clearance_mm)
        model, meta = cpr.train_diffusionnet_regressor(
            fit_s, epochs=epochs, device=device, op_cache_dir=op_cache_dir,
            log_every=max(1, epochs // 10), max_gpu_verts=max_gpu_verts,
            metrics_fn=metrics_fn, eval_every=eval_every, patience=patience,
            best_path=best_path, last_path=last_path, save_every=save_every,
            history_path=history_path, resume_from=resume_from, seed=seed,
            heat_pos_weight=heat_pos_weight, w_heat=w_heat, w_off=w_off, w_dir=w_dir,
            heat_loss=heat_loss, focal_gamma=focal_gamma,
            lr_decay_every=lr_decay_every, lr_decay_rate=lr_decay_rate,
            accum_steps=accum_steps, low_memory=low_memory,
            decode=decode, train_config=train_config, best_metric=best_metric,
            resume_strict=resume_strict, snapshot_every=snapshot_every,
            weight_decay=weight_decay, lr_schedule=lr_schedule,
            warmup_epochs=warmup_epochs, grad_clip=grad_clip,
            offset_heat_weight=offset_heat_weight, part_weight_mode=part_weight_mode)
        infer_arr = _infer_builder(model, meta)

    train_reports = evaluate(train_s, train_m, infer_arr,
                             dist_thresh_mm=dist_thresh_mm,
                             heatmap_thresh=heatmap_thresh, min_votes=min_votes,
                             nms_clearance_mm=nms_clearance_mm)
    if print_points and val_s:
        print("\n=== VAL per-part metrics (accuracy / precision / recall / F1) ===")
    val_reports = (evaluate(val_s, val_m, infer_arr,
                            dist_thresh_mm=dist_thresh_mm,
                            heatmap_thresh=heatmap_thresh, min_votes=min_votes,
                            nms_clearance_mm=nms_clearance_mm,
                            print_points=print_points)
                   if val_s else [])
    # frozen held-out TEST split: one unbiased number, scored at the deploy
    # decode point. Never touched training or best-checkpoint/decode selection.
    test_reports = (evaluate(test_s, test_m, infer_arr,
                             dist_thresh_mm=dist_thresh_mm,
                             heatmap_thresh=heatmap_thresh, min_votes=min_votes,
                             nms_clearance_mm=nms_clearance_mm)
                    if test_s else [])
    return (aggregate(train_reports), aggregate(val_reports), train_reports,
            val_reports, aggregate(test_reports), test_reports)


def _print_agg(title, agg):
    """Readable metric block: per-part means + pooled (micro) values."""
    print(f"\n=== {title} ===")
    if not agg:
        print("  (no parts)")
        return
    print("  parts=%d  GT=%d  TP=%d  FP=%d  FN=%d"
          % (agg["n_parts"], agg["total_gt"], agg["total_tp"],
             agg["total_fp"], agg["total_fn"]))
    for k in ("accuracy", "precision", "recall", "f1"):
        print("  %-9s : %.2f%%   (pooled %.2f%%)"
              % (k, agg[k] * 100, agg["micro_" + k] * 100))
    print("  loc error : %.3f mm    ang error : %.2f deg"
          % (agg["mean_loc_err_mm"], agg["mean_ang_err_deg"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train connection-point detector on ABB JSON data")
    ap.add_argument("source", help="corpus: a directory, a single big JSON array, or one part file")
    ap.add_argument("--backbone", choices=["mlp", "diffusionnet", "knngraph"],
                    default="mlp")
    ap.add_argument("--epochs", type=int, default=300,
                    help="TOTAL epochs to reach; with --resume, training continues "
                         "from the checkpoint's epoch up to this target")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--dist-thresh-mm", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-cache-dir", default=None,
                    help="cache dir for DiffusionNet mesh operators "
                         "(strongly recommended for the 479-file corpus)")
    ap.add_argument("--max-gpu-verts", type=int, default=60000,
                    help="parts larger than this run on CPU even with --device cuda "
                         "(avoids OOM on small GPUs); 0 forces GPU for all parts")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="directory for the .ckpt files (default: checkpoints/, "
                         "tracked in git so other machines get them on clone/pull)")
    ap.add_argument("--run-name", default=None,
                    help="basename for the checkpoint files "
                         "(default: cp_<backbone>)")
    ap.add_argument("--resume", action="store_true",
                    help="continue training from <ckpt-dir>/<run-name>_last.ckpt "
                         "(falls back to _best.ckpt); restores model+optimizer+"
                         "scheduler+epoch, so a run can hop between machines")
    ap.add_argument("--resume-from", default=None,
                    help="explicit checkpoint path to resume from (overrides --resume)")
    ap.add_argument("--ckpt", default=None,
                    help="explicit path for the BEST checkpoint "
                         "(default <ckpt-dir>/<run-name>_best.ckpt); reload with "
                         "cp_regressor.load_model")
    ap.add_argument("--ckpt-every", type=int, default=1,
                    help="write the resumable last.ckpt every N epochs "
                         "(default 1; 0 = only at the end)")
    ap.add_argument("--print-points", action="store_true",
                    help="after training, print per-part keypoint metrics for the "
                         "validation parts: accuracy / precision / recall / F1 plus "
                         "TP/FP/FN and mean loc/ang error, one line per part")
    ap.add_argument("--out", default=None,
                    help="write aggregate + per-part metrics (with timing) to this JSON file")
    ap.add_argument("--export-model", default=None,
                    help="after training, write a lean inference-only .pt from the "
                         "best checkpoint (weights+meta+backbone+decode params, no "
                         "optimizer/history). Load it with predict.py")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="min voting vertices per decoded CP. Default 1: the GT-decode "
                         "ceiling on the real corpus is R=0.94 at min_votes=1 vs only "
                         "0.79 at 2 (diag_encoding.py), so >1 throws away recoverable "
                         "CPs AND biases best-checkpoint selection. Raise it only via "
                         "sweep_decode if precision needs it")
    ap.add_argument("--heatmap-thresh", type=float, default=0.3,
                    help="decode: min sigmoid heat for a vertex to vote for a CP. "
                         "Higher -> fewer false positives, but too high fires "
                         "nothing. This is just the training-time eval point; lock "
                         "the deployment operating point with sweep_decode.py")
    ap.add_argument("--nms-clearance-mm", type=float, default=5.0,
                    help="decode: votes are merged within max(2*sigma, this) mm; "
                         "larger collapses scattered votes into fewer detections")
    ap.add_argument("--heat-pos-weight", type=float, default=50.0,
                    help="heatmap BCE positive weight (1 + w*h). The heat target is "
                         "~99%% background, so this stays high (default 50, known-good) "
                         "or the model collapses to ~0 heat everywhere (recall 0). "
                         "Lower it (e.g. 10-15) only if the heatmap floods/over-fires "
                         "at a FULL epoch budget (knngraph/diffusionnet only)")
    ap.add_argument("--heat-loss", choices=["bce", "focal", "centernet"],
                    default="centernet",
                    help="heatmap loss (DEFAULT centernet -- bce/focal flood or "
                         "collapse on this ~99%%-background heatmap). 'centernet' is "
                         "the penalty-reduced focal "
                         "loss normalised by #keypoints -- the right choice for this "
                         "sparse heatmap; 'bce'/'focal' average over all vertices and "
                         "either over-fire or collapse on the 99%% background")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--lr-decay-every", type=int, default=0,
                    help="StepLR step size in epochs (0 disables the schedule)")
    ap.add_argument("--lr-decay-rate", type=float, default=0.5)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation over N parts before opt.step")
    ap.add_argument("--low-memory", action="store_true",
                    help="reload eigenbasis from cache per step instead of holding "
                         "all in RAM (for corpora larger than memory)")
    ap.add_argument("--knn-k", type=int, default=16,
                    help="knngraph: neighbours per vertex in the kNN graph (default 16). "
                         "Larger k = wider receptive field per layer but slower/more "
                         "memory; sweep 12-24")
    ap.add_argument("--knn-width", type=int, default=128,
                    help="knngraph: EdgeConv feature width (default 128)")
    ap.add_argument("--knn-layers", type=int, default=4,
                    help="knngraph: number of residual EdgeConv blocks (default 4)")
    ap.add_argument("--knn-global", action="store_true",
                    help="knngraph: add a PointNet-style global-context feature "
                         "(global max-pool concatenated before the head). Helps the "
                         "model judge a vertex relative to the whole part -- usually a "
                         "precision/recall win for these sparse connection points")
    ap.add_argument("--augment", type=int, default=0,
                    help="add N augmented clones per TRAINING part (0=off). Each is a "
                         "random rigid rotation (+ optional jitter) with CP points/"
                         "directions transformed consistently. The main generalisation "
                         "lever for mlp/knngraph (raw xyz -> no pose invariance). Try 4-8. "
                         "NOTE: with --backbone diffusionnet each clone is fresh geometry, "
                         "so the eigenbasis cache grows ~Nx and precompute takes ~Nx longer")
    ap.add_argument("--no-aug-rotate", dest="aug_rotate", action="store_false",
                    help="disable the random rotation in --augment (keep only jitter)")
    ap.add_argument("--aug-jitter-frac", type=float, default=0.0,
                    help="augmentation vertex jitter: Gaussian std as a fraction of the "
                         "bbox diagonal, added to vertices only (GT CPs stay). ~0.005 "
                         "simulates mesh/scan noise; 0 disables jitter")
    ap.add_argument("--aug-reflect", action="store_true",
                    help="augmentation: also mirror each clone across a random axis with "
                         "p=0.5 (connector geometry is often mirror-symmetric). CP points/"
                         "directions are mirrored too and face winding is flipped")
    ap.set_defaults(aug_rotate=True)
    ap.add_argument("--split-group", choices=["none", "prefix", "geometry"],
                    default="none",
                    help="how to group parts for a LEAKAGE-SAFE train/val split: 'none' "
                         "(per-PartNr, old default), 'prefix' (group part-family variants "
                         "by stripped PartNr), 'geometry' (group near-identical meshes). "
                         "Stops sibling variants inflating val F1 by leaking across splits")
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="carve a frozen held-out TEST split (fraction). It never trains "
                         "and never tunes decode/checkpoints -- one unbiased final number "
                         "reported alongside train/val. 0 = no test split")
    ap.add_argument("--cp-surface-tol-frac", type=float, default=None,
                    help="drop any GT connection point farther than this fraction of the "
                         "bbox diagonal from the nearest vertex (off-surface mislabels). "
                         "Unset = keep all, only warn at 5%%")
    ap.add_argument("--neg-frac", type=float, default=0.0,
                    help="include this fraction of the train-side ZERO-CP parts as "
                         "negatives in the fit set, so the model learns to stay silent "
                         "where there is nothing (reduces false positives). 0 = drop them")
    ap.add_argument("--max-neg", type=int, default=None,
                    help="cap on the number of negative (zero-CP) parts added by --neg-frac")
    ap.add_argument("--best-metric", choices=["micro_f1", "f1"], default="micro_f1",
                    help="metric the best checkpoint is selected on. Default micro_f1 "
                         "(pooled -- the metric sweeps/reports/deployment use); 'f1' is "
                         "the per-part macro mean (the old behaviour)")
    ap.add_argument("--strict-resume", dest="resume_strict", action="store_true",
                    help="on --resume, REFUSE to continue if a critical hyperparameter "
                         "(val_frac/seed/loss/augment/...) differs from the checkpoint, "
                         "instead of just warning (best_f1/history would be incomparable)")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also write a never-overwritten <run>_ep<N>.ckpt every N epochs, "
                         "so an earlier good model is recoverable if a later best overfits "
                         "(0 = off; only best+last are kept)")
    # --- loss design ---
    ap.add_argument("--w-heat", type=float, default=1.0,
                    help="loss weight on the heatmap term (centernet). Sweep the "
                         "w-heat/w-off/w-dir balance -- it drives precision/recall vs "
                         "localisation error")
    ap.add_argument("--w-off", type=float, default=5.0,
                    help="loss weight on the offset (localisation) L1 term")
    ap.add_argument("--w-dir", type=float, default=1.0,
                    help="loss weight on the direction cosine term")
    ap.add_argument("--no-offset-heat-weight", dest="offset_heat_weight",
                    action="store_false",
                    help="disable weighting the offset/dir loss by target heat (by "
                         "default the vertices that actually vote get the most accurate "
                         "offsets)")
    ap.set_defaults(offset_heat_weight=True)
    ap.add_argument("--part-weight", choices=["none", "keypoints"], default="none",
                    help="weight each part's loss by its #connection-points ('keypoints') "
                         "so multi-CP parts are not down-weighted to a 1-CP part (default "
                         "none)")
    # --- regularisation / optimisation ---
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay (L2). Try 1e-4..1e-2 on this small corpus to "
                         "fight overfitting (0 = plain Adam)")
    ap.add_argument("--knn-dropout", type=float, default=0.0,
                    help="knngraph: dropout before the output head (0 = off; try 0.1-0.3)")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="clip gradient L2 norm before opt.step (0 = off; ~1-5 tames the "
                         "occasional large EdgeConv gradient with batch=1)")
    ap.add_argument("--lr-schedule", choices=["step", "plateau", "cosine"],
                    default="step",
                    help="LR schedule: step (StepLR via --lr-decay-every), plateau "
                         "(ReduceLROnPlateau on val F1), or cosine. Add --warmup-epochs "
                         "for a linear warmup")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="linear LR warmup over this many epochs before the main schedule")
    ap.add_argument("--amp", action="store_true",
                    help="knngraph+cuda: enable mixed precision (fp16 autocast + "
                         "GradScaler). OFF by default and only worth it at a very high "
                         "--max-gpu-verts (>=~16k): EdgeConv is bandwidth-bound, so "
                         "below the memory cliff fp16 overhead makes it SLOWER. Prefer "
                         "lowering --max-gpu-verts (8000) for speed instead")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="run val eval + best-checkpoint every N epochs (diffusionnet)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many stale val evals (0 disables)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # checkpoint file layout: best (by val F1), last (resume point), history
    run_name = args.run_name or f"cp_{args.backbone}"
    if args.ckpt:
        best_path = args.ckpt
        stem = os.path.splitext(args.ckpt)[0]
        last_path = stem + "_last.ckpt"
        history_path = stem + "_history.json"
    else:
        best_path = os.path.join(args.ckpt_dir, run_name + "_best.ckpt")
        last_path = os.path.join(args.ckpt_dir, run_name + "_last.ckpt")
        history_path = os.path.join(args.ckpt_dir, run_name + "_history.json")

    resume_from = args.resume_from
    if resume_from is None and args.resume:
        if os.path.exists(last_path):
            resume_from = last_path
        elif os.path.exists(best_path):
            resume_from = best_path
        else:
            logger.warning("--resume: no checkpoint at %s or %s -- starting fresh",
                           last_path, best_path)
    if resume_from:
        logger.info("resuming from %s", resume_from)

    import time
    t0 = time.time()
    tr, va, tr_reps, va_reps, te, te_reps = train_and_eval(
        args.source, backbone=args.backbone, epochs=args.epochs,
        val_frac=args.val_frac, max_parts=args.max_parts,
        device=args.device, dist_thresh_mm=args.dist_thresh_mm, seed=args.seed,
        op_cache_dir=args.op_cache_dir, max_gpu_verts=args.max_gpu_verts,
        best_path=best_path, last_path=last_path, history_path=history_path,
        save_every=args.ckpt_every, resume_from=resume_from,
        min_votes=args.min_votes, heatmap_thresh=args.heatmap_thresh,
        nms_clearance_mm=args.nms_clearance_mm, heat_pos_weight=args.heat_pos_weight,
        heat_loss=args.heat_loss, focal_gamma=args.focal_gamma,
        lr_decay_every=args.lr_decay_every, lr_decay_rate=args.lr_decay_rate,
        accum_steps=args.accum_steps, low_memory=args.low_memory,
        eval_every=args.eval_every, patience=args.patience,
        augment=args.augment, aug_rotate=args.aug_rotate,
        aug_jitter_frac=args.aug_jitter_frac, aug_reflect=args.aug_reflect,
        split_group=args.split_group, test_frac=args.test_frac,
        cp_surface_tol_frac=args.cp_surface_tol_frac,
        neg_frac=args.neg_frac, max_neg=args.max_neg,
        best_metric=args.best_metric, resume_strict=args.resume_strict,
        snapshot_every=args.snapshot_every,
        w_heat=args.w_heat, w_off=args.w_off, w_dir=args.w_dir,
        weight_decay=args.weight_decay, lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs, grad_clip=args.grad_clip,
        offset_heat_weight=args.offset_heat_weight, part_weight_mode=args.part_weight,
        knn_config={"k": args.knn_k, "c_width": args.knn_width,
                    "n_layers": args.knn_layers, "global_feat": args.knn_global,
                    "dropout": args.knn_dropout},
        print_points=args.print_points, amp=args.amp)
    elapsed = time.time() - t0
    _print_agg("TRAIN metrics", tr)
    _print_agg("VAL metrics", va)
    if te:
        _print_agg("TEST metrics (held-out, never tuned)", te)
    print("\nelapsed: %.0f s (%.2f h)" % (elapsed, elapsed / 3600))

    print("\ncheckpoints:")
    for tag, p in (("best", best_path), ("last", last_path),
                   ("history", history_path)):
        print("  %-7s %s%s" % (tag, p,
                               "" if os.path.exists(p) else "  (not written)"))
    print("\nto continue this run later (same or another machine):")
    print("  # checkpoints are BINARY -- share via Git LFS or a drive/artifact store,")
    print("  # NOT plain `git add` (every epoch rewrites last.ckpt and bloats history).")
    print("  # one-time LFS setup:  git lfs install && git lfs track '*.ckpt'")
    print("  #   (commit .gitattributes; then add/commit/push the .ckpt as usual)")
    print("  # then on the other machine: pull the checkpoints and run:")
    print("  python train_cp.py %s --backbone %s --resume --epochs <target>"
          % (args.source, args.backbone))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"train_aggregate": tr, "val_aggregate": va,
                       "test_aggregate": te,
                       "train_per_part": tr_reps, "val_per_part": va_reps,
                       "test_per_part": te_reps,
                       "elapsed_sec": elapsed}, fh, indent=2)
        print("\nmetrics written to", args.out)

    if args.export_model:
        # bundle the decode operating point used for this run so the exported
        # model is self-contained (the weights alone are ambiguous for keypoints)
        decode = {"heatmap_thresh": args.heatmap_thresh, "min_votes": args.min_votes,
                  "nms_clearance_mm": args.nms_clearance_mm,
                  "dist_thresh_mm": args.dist_thresh_mm}
        src = best_path if os.path.exists(best_path) else last_path
        if os.path.exists(src):
            cpr.export_inference_checkpoint(src, args.export_model, decode=decode)
            print("\nexported inference model -> %s\n  (from %s, decode=%s)"
                  % (args.export_model, src, decode))
            print("  run it:  python predict.py %s <mesh-or-corpus> --out preds.json"
                  % args.export_model)
        else:
            print("\n--export-model: no checkpoint found to export (%s)" % src)


if __name__ == "__main__":
    main()

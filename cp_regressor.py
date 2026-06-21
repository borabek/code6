# Connection-point regressor (Option B): per-vertex 7-channel head + losses +
# training loop. Predicts (heatmap, offset xyz, direction xyz) per vertex; the
# decoder in cp_targets turns that into discrete terminal-block detections.
#
# Two backbones are provided:
#   * build_diffusionnet_regressor() – the production head: a DiffusionNet with
#     C_out=7 (reuses diffusionnet.precompute_operators / _forward). Needs the
#     `diffusion_net` package.
#   * CPMLP – a light per-vertex MLP on normalised xyz that needs only torch.
#     Used for smoke-tests / environments without the diffusion_net compile
#     chain. It has no geometric receptive field, so it is NOT the production
#     model, but it exercises the full target/loss/decode pipeline.
#
# torch is imported lazily (like diffusionnet.py) so the data utilities remain
# importable without it.

import hashlib
import logging
import numpy as np

import cp_targets as ct

logger = logging.getLogger(__name__)

N_CHANNELS = ct.N_CHANNELS  # 7


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError("'torch' is required for cp_regressor "
                          "(pip install torch)") from exc


# ---------------------------------------------------------------------------
# normalisation: center + scale vertices so the MLP/optimiser see ~unit coords
# ---------------------------------------------------------------------------

def normalize_vertices(V):
    """Center on centroid and scale by bounding-box diagonal. Returns (Vn, center, scale)."""
    V = np.asarray(V, dtype=np.float64)
    center = V.mean(0)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    scale = diag if diag > 0 else 1.0
    return (V - center) / scale, center, scale


# ---------------------------------------------------------------------------
# light MLP backbone (smoke-test / no diffusion_net)
# ---------------------------------------------------------------------------

def build_cpmlp(width=128, depth=4, c_in=3):
    """Per-vertex MLP: c_in -> ... -> 7. torch.nn.Module."""
    _require_torch()
    import torch.nn as nn

    class CPMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(c_in, width), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.ReLU()]
            layers += [nn.Linear(width, N_CHANNELS)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return CPMLP()


# ---------------------------------------------------------------------------
# production backbone: DiffusionNet with a 7-channel regression head
# ---------------------------------------------------------------------------

def build_diffusionnet_regressor(config=None):
    """Build a DiffusionNet with C_out=7 (regression). Returns (model, meta)."""
    import diffusionnet as dnmod
    _require_torch()
    dn = dnmod._require("diffusion_net", "pip install git+https://github.com/nmwsharp/diffusion-net")
    cfg = {**dnmod.DEFAULTS, **(config or {})}
    c_in = 3 if cfg.get("input_features", "xyz") == "xyz" else 16
    model = dn.layers.DiffusionNet(
        C_in=c_in, C_out=N_CHANNELS,
        C_width=int(cfg.get("c_width", 128)),
        N_block=int(cfg.get("n_diffusion_blocks", 4)),
        last_activation=None,            # raw outputs; we apply sigmoid/normalise in post
        outputs_at="vertices",
        dropout=float(cfg.get("dropout", 0.0)) > 0.0,
    )
    meta = {"input_features": cfg.get("input_features", "xyz"),
            "k_eig": int(cfg.get("n_eig", 128)), "hks_count": 16,
            # structural params -> a checkpoint is self-describing and rebuildable
            "c_in": c_in, "c_width": int(cfg.get("c_width", 128)),
            "n_block": int(cfg.get("n_diffusion_blocks", 4)),
            "dropout": float(cfg.get("dropout", 0.0))}
    return model, meta


# ---------------------------------------------------------------------------
# checkpointing: one .ckpt format for ALL backbones, with full training state
# ---------------------------------------------------------------------------
# A checkpoint is a dict with at least {state_dict, meta, backbone}. meta
# carries the structural params so the exact architecture can be rebuilt
# before loading weights -- no separate config needed for inference later.
#
# Training checkpoints additionally carry {optimizer, scheduler, epoch,
# best_f1, history, rng} so a run can RESUME exactly where it stopped --
# including on a different machine: commit/push the .ckpt, pull it on the
# other device, and run train_cp.py --resume. torch.save files are portable
# across OS/CPU/GPU (tensors are remapped via map_location on load).

CKPT_VERSION = 1

# Bumped whenever the encode/decode SEMANTICS change (target channel meaning,
# sigma rule, vote/NMS math) so an old exported .pt that would now mis-decode is
# flagged at load instead of silently producing wrong points. Independent of
# CKPT_VERSION, which tracks the checkpoint container format.
# v2: sigma capped to <=0.5x closest CP spacing in encode_targets; decode NMS
#     radius decoupled from sigma (fixed tool clearance) so close CPs resolve.
DECODE_SCHEMA = 2

# Decode operating point bundled with an inference artifact. For keypoint
# detection these three knobs are part of "the model": the same weights yield a
# very different precision/recall depending on them, so an export carries them
# explicitly. dist_thresh_mm is the match radius used when scoring, kept for
# provenance. Older checkpoints without a "decode" key fall back to these.
# min_votes=1 matches the deploy default everywhere else (the GT-decode ceiling
# is R=0.94 @1 vs 0.79 @2 on the real corpus -- see diag_encoding.py).
DEFAULT_DECODE = {"heatmap_thresh": 0.3, "min_votes": 1,
                  "nms_clearance_mm": 5.0, "dist_thresh_mm": 5.0}


# ---------------------------------------------------------------------------
# vertex subsampling -- ONE strategy shared by training and inference
# ---------------------------------------------------------------------------
# A part above the model's vertex cap is uniformly subsampled to that cap so the
# kNN graph keeps the density the model trained on. CRITICAL: inference is
# GT-free, so it CANNOT keep CP-peak vertices -- therefore TRAINING must use the
# same plain-uniform sample (and re-snap its heat peaks onto the kept vertices),
# otherwise the model trains on a vertex distribution inference never reproduces
# and recall silently drops on big parts.

def uniform_subsample_idx(n, cap, seed=0):
    """Deterministic uniform subsample: sorted indices of `cap` of `n` vertices.
    Same routine on both sides so train and inference match in distribution."""
    if not cap or n <= cap:
        return None
    return np.sort(np.random.RandomState(seed).choice(n, cap, replace=False))


def subsample_coverage(verts_norm, gt_points, center, scale, cap, seed=0,
                       sigma_frac=0.02):
    """How many GT points become UNRECOVERABLE under the inference subsample.

    A GT point whose nearest *kept* vertex is farther than ~3 sigma gets ~0 heat
    everywhere, so no vertex can vote for it -> a guaranteed false negative that
    is purely an artefact of subsampling. Returns (n_gt, n_at_risk). GT-aware, so
    for diagnostics only -- never feeds decode."""
    n = len(verts_norm)
    if not cap or n <= cap or len(gt_points) == 0:
        return len(gt_points), 0
    idx = uniform_subsample_idx(n, cap, seed)
    kept = np.asarray(verts_norm)[idx]
    gtn = (np.asarray(gt_points, dtype=np.float64) - center) / scale
    thr = 3.0 * sigma_frac                      # normalised diag ~= 1
    at_risk = 0
    for g in gtn:
        if np.linalg.norm(kept - g, axis=1).min() > thr:
            at_risk += 1
    return len(gt_points), at_risk


def _meta_to_config(backbone, meta):
    """Map a checkpoint's meta back to the builder config for that backbone."""
    if backbone == "diffusionnet":
        return {"input_features": meta["input_features"], "c_width": meta["c_width"],
                "n_diffusion_blocks": meta["n_block"], "n_eig": meta["k_eig"],
                "dropout": meta.get("dropout", 0.0)}
    if backbone == "knngraph":
        return {"c_width": meta["c_width"], "n_layers": meta["n_layers"],
                "k": meta["k"], "global_feat": meta.get("global_feat", False),
                "dropout": meta.get("dropout", 0.0)}
    if backbone == "mlp":
        return {"width": meta.get("width", 128), "depth": meta.get("depth", 4)}
    raise ValueError(f"unknown backbone {backbone!r}")


def build_regressor(backbone, config=None):
    """Build any of the three backbones. Returns (model, meta)."""
    if backbone == "diffusionnet":
        return build_diffusionnet_regressor(config)
    if backbone == "knngraph":
        return build_knngraph_regressor(config)
    if backbone == "mlp":
        cfg = {"width": 128, "depth": 4, **(config or {})}
        model = build_cpmlp(width=int(cfg["width"]), depth=int(cfg["depth"]))
        meta = {"backbone": "mlp", "input_features": "xyz", "c_in": 3,
                "width": int(cfg["width"]), "depth": int(cfg["depth"])}
        return model, meta
    raise ValueError(f"unknown backbone {backbone!r}")


def _capture_rng():
    """Snapshot python/numpy/torch(/cuda) RNG states for exact resume."""
    torch = _require_torch()
    import random
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(st):
    """Best-effort RNG restore (a resume on different hardware still works --
    it just reshuffles from the seed instead of the exact saved stream)."""
    if not st:
        return
    torch = _require_torch()
    import random
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"].cpu().to(torch.uint8))
        cuda = st.get("cuda")
        if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in cuda])
    except Exception as exc:                           # noqa: BLE001
        logger.warning("could not restore RNG state (%s) -- continuing", exc)


# cap on how many per-eval history records are EMBEDDED in a checkpoint. The full
# history lives in the sibling history JSON (history_path); the checkpoint keeps
# only a tail for self-containment so per-epoch saves do not grow without bound.
CKPT_HISTORY_CAP = 200


def save_checkpoint(path, model, meta, backbone, optimizer=None, scheduler=None,
                    epoch=None, best_f1=None, history=None, decode=None,
                    train_config=None, best_metric=None, quiet=False):
    """Write a (resumable) checkpoint atomically.

    With only model+meta this is an inference checkpoint (what the legacy
    save_diffusionnet/save_knngraph wrote); passing optimizer/scheduler/epoch
    makes it a full training checkpoint that train loops can resume from.
    `epoch` is the last COMPLETED epoch (0-based). `decode` (a dict of the
    operating-point knobs) is stored when given so the artifact is unambiguous.
    `train_config` (val_frac/seed/loss/augment/...) is stored so a --resume can
    detect a mismatched continuation. Embedded history is capped (full log lives
    in the history JSON). Atomic write (tmp+replace) so a crash mid-save never
    corrupts an existing checkpoint.
    """
    torch = _require_torch()
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ckpt = {"ckpt_version": CKPT_VERSION, "decode_schema": DECODE_SCHEMA,
            "backbone": backbone, "state_dict": model.state_dict(), "meta": meta}
    if decode is not None:
        ckpt["decode"] = dict(decode)
    if train_config is not None:
        ckpt["train_config"] = dict(train_config)
    if best_metric is not None:
        ckpt["best_metric"] = str(best_metric)
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
        ckpt["rng"] = _capture_rng()
    if best_f1 is not None:
        ckpt["best_f1"] = float(best_f1)
    if history is not None:
        ckpt["history"] = list(history)[-CKPT_HISTORY_CAP:]
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    # quiet: the rolling per-epoch 'last' snapshot would otherwise spam one INFO
    # line every epoch and drown out the train-loss/val output.
    (logger.debug if quiet else logger.info)(
        "saved %s checkpoint -> %s%s", backbone, path,
        f" (epoch {epoch})" if epoch is not None else "")


_SAFE_GLOBALS_DONE = False


def _allow_numpy_safe_globals(torch):
    """Allowlist the numpy unpickling primitives that appear in our `meta` dicts
    (plain arrays/dtypes) so a weights_only=True load SUCCEEDS for inference
    checkpoints instead of falling back to the unsafe full load. These globals
    reconstruct numpy arrays only -- they execute no arbitrary code."""
    global _SAFE_GLOBALS_DONE
    if _SAFE_GLOBALS_DONE:
        return
    try:
        allow = [np.ndarray, np.dtype]
        for modname in ("numpy._core.multiarray", "numpy.core.multiarray"):
            try:
                m = __import__(modname, fromlist=["_reconstruct", "scalar"])
                allow.append(m._reconstruct)
                if hasattr(m, "scalar"):
                    allow.append(m.scalar)
            except Exception:                                # noqa: BLE001
                pass
        # every concrete numpy dtype class (Float64DType, UInt32DType, ...) plus
        # the scalar types -- all pure data reconstructors, no code execution.
        try:
            import numpy.dtypes as _ndt
            allow += [getattr(_ndt, n) for n in dir(_ndt)
                      if isinstance(getattr(_ndt, n), type)]
        except Exception:                                    # noqa: BLE001
            pass
        for nm in ("float64", "float32", "float16", "int64", "int32", "int16",
                   "int8", "uint64", "uint32", "uint16", "uint8", "bool_"):
            if hasattr(np, nm):
                allow.append(getattr(np, nm))
        torch.serialization.add_safe_globals(allow)
    except Exception:                                        # noqa: BLE001
        pass
    _SAFE_GLOBALS_DONE = True


def load_checkpoint(path, device="cpu", weights_only=False):
    """Load a checkpoint dict (any backbone, training or inference-only).

    weights_only=True is the SAFE load for deployment: it refuses to unpickle
    arbitrary code (these .ckpt travel between machines via git, so a tampered
    file is a real supply-chain risk). It works for inference checkpoints
    (weights + plain-dict/numpy meta/decode) but not for resume checkpoints whose
    optimizer/scheduler state needs full unpickling -- those fall back to False.
    """
    torch = _require_torch()
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if weights_only:
        _allow_numpy_safe_globals(torch)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=weights_only)
    except Exception as exc:                                  # noqa: BLE001
        if not weights_only:
            raise
        logger.warning("safe (weights_only) load of %s failed (%s) -- falling back "
                       "to a full load; only do this for checkpoints you trust", path, exc)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ckpt or "meta" not in ckpt:
        raise ValueError(f"{path} is not a cp_regressor checkpoint")
    v = ckpt.get("ckpt_version")
    if v is not None and v != CKPT_VERSION:
        logger.warning("checkpoint %s is format version %s but this code expects %d "
                       "-- regenerate/re-export it if loading fails", path, v, CKPT_VERSION)
    return ckpt


def _model_from_ckpt(ckpt, device="cpu"):
    """Rebuild the exact architecture from a loaded checkpoint dict and load its
    weights. Returns (model on `device`, in eval mode; meta; backbone)."""
    backbone = ckpt.get("backbone") or ckpt["meta"].get("backbone", "diffusionnet")
    model, _ = build_regressor(backbone, _meta_to_config(backbone, ckpt["meta"]))
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        # The model definition changed since this checkpoint was written (e.g. the
        # knngraph EdgeConv gained LayerNorm), so the state_dict keys no longer line
        # up. Surface that plainly instead of a wall of "Missing/Unexpected key(s)".
        raise RuntimeError(
            f"'{backbone}' checkpoint is from an OLDER architecture and can't be "
            f"loaded by the current code -- retrain it, or check out the matching "
            f"code revision. (state_dict mismatch: {str(exc).splitlines()[0]})"
        ) from exc
    model = model.to(device)
    model.eval()
    return model, ckpt["meta"], backbone


def load_model(path, device="cpu"):
    """Rebuild a trained regressor from any checkpoint for inference.

    Works for all backbones and both old ({state_dict, meta}) and new
    full-training checkpoints. Returns (model, meta, backbone); model is on
    `device` and in eval mode.
    """
    ckpt = load_checkpoint(path, device=device, weights_only=True)
    model, meta, backbone = _model_from_ckpt(ckpt, device=device)
    logger.info("loaded %s checkpoint <- %s", backbone, path)
    return model, meta, backbone


def load_inference(path, device="cpu"):
    """Like load_model, but also returns the decode operating point bundled with
    the artifact. Returns (model, meta, backbone, decode) where decode is a dict
    with heatmap_thresh / min_votes / nms_clearance_mm / dist_thresh_mm --
    DEFAULT_DECODE for older checkpoints that predate the bundled params.
    """
    ckpt = load_checkpoint(path, device=device, weights_only=True)
    model, meta, backbone = _model_from_ckpt(ckpt, device=device)
    decode = {**DEFAULT_DECODE, **(ckpt.get("decode") or {})}
    # warn (don't crash) when the artifact predates / postdates the current
    # encode/decode semantics, so a stale .pt that would mis-decode is visible.
    schema = ckpt.get("decode_schema")
    if schema is None:
        logger.warning("model %s has no decode_schema (pre-versioning export) -- "
                       "verify it was trained with the current encode/decode before "
                       "trusting its points", path)
    elif schema != DECODE_SCHEMA:
        logger.warning("model %s decode_schema=%s != current %d -- encode/decode "
                       "semantics changed; re-export or re-train this model",
                       path, schema, DECODE_SCHEMA)
    logger.info("loaded %s checkpoint <- %s (decode=%s)", backbone, path, decode)
    return model, meta, backbone, decode


def export_inference_checkpoint(src_path, dst_path, decode=None, device="cpu"):
    """Write a lean inference-only artifact from a (possibly fat training)
    checkpoint.

    Keeps weights + meta + backbone (+ decode operating point); drops the
    optimizer/scheduler/epoch/RNG/history a resumable checkpoint carries. This
    roughly halves the file and makes it an unambiguous deployment model. If
    `decode` is None the source checkpoint's own decode (if any) is preserved.
    Atomic write. Returns dst_path.
    """
    torch = _require_torch()
    import os
    ckpt = load_checkpoint(src_path, device=device, weights_only=True)
    backbone = ckpt.get("backbone") or ckpt["meta"].get("backbone", "diffusionnet")
    lean = {"ckpt_version": CKPT_VERSION, "decode_schema": DECODE_SCHEMA,
            "backbone": backbone,
            "state_dict": ckpt["state_dict"], "meta": ckpt["meta"]}
    if decode is not None:
        lean["decode"] = dict(decode)
    elif ckpt.get("decode"):
        lean["decode"] = dict(ckpt["decode"])
    d = os.path.dirname(os.path.abspath(dst_path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = dst_path + ".tmp"
    torch.save(lean, tmp)
    os.replace(tmp, dst_path)
    logger.info("exported lean %s inference model -> %s%s", backbone, dst_path,
                f" (decode={lean['decode']})" if "decode" in lean else "")
    return dst_path


# Backward-compatible wrappers (the readme and older scripts reference these).

def save_diffusionnet(model, meta, path):
    """Save a trained DiffusionNet regressor (weights + meta) to `path`."""
    save_checkpoint(path, model, meta, "diffusionnet")


def load_diffusionnet(path, device="cpu"):
    """Rebuild a DiffusionNet regressor from a checkpoint. Returns (model, meta)."""
    model, meta, _ = load_model(path, device=device)
    return model, meta


# ---------------------------------------------------------------------------
# shared training plumbing: setup/resume + per-epoch bookkeeping
# ---------------------------------------------------------------------------

# train_config keys that MUST match for a resume to be coherent. heat_pos_weight
# changes the loss; val_frac/seed/split_group change the val set best_f1 is
# measured on; best_metric changes what best_f1 even means; augment changes the
# fit set. Continuing across a change to any of these compares apples to oranges.
_RESUME_CRITICAL = ("val_frac", "seed", "split_group", "heat_pos_weight",
                    "heat_loss", "focal_gamma", "augment", "best_metric",
                    "max_gpu_verts")


def _check_resume_config(stored, current, strict):
    """Warn (or, if strict, raise) when a --resume changes a critical hyperparam,
    so a continuation never silently compares against an incomparable best_f1."""
    if not stored:
        logger.warning("resume: checkpoint has no stored train_config -- cannot "
                       "verify the continuation matches; proceeding")
        return
    diffs = []
    for k in _RESUME_CRITICAL:
        if k in stored and k in current and stored[k] != current[k]:
            diffs.append("%s %r->%r" % (k, stored[k], current[k]))
    if diffs:
        msg = "resume: training config changed vs the checkpoint (" + \
              "; ".join(diffs) + ") -- best_f1/history are no longer comparable"
        if strict:
            raise ValueError(msg + " [--strict-resume]")
        logger.warning(msg + " -- continuing anyway (pass --strict-resume to refuse)")


def _build_scheduler(opt, lr_schedule, lr_decay_every, lr_decay_rate,
                     warmup_epochs, total_epochs):
    """Return (scheduler, is_metric_driven). 'step' = StepLR (the default);
    'plateau' = ReduceLROnPlateau on the val metric (stepped by the bookkeeper);
    'cosine' = cosine decay, both with an optional linear warmup prepended."""
    torch = _require_torch()
    sl = torch.optim.lr_scheduler
    warm = int(warmup_epochs or 0)
    if lr_schedule == "plateau":
        # mode='max': we maximise val F1; patience/factor are gentle defaults
        return sl.ReduceLROnPlateau(opt, mode="max", factor=float(lr_decay_rate),
                                    patience=max(1, int(lr_decay_every or 5))), True
    if lr_schedule == "cosine":
        T = max(1, int(total_epochs or 100) - warm)
        cos = sl.CosineAnnealingLR(opt, T_max=T)
        if warm:
            wu = sl.LinearLR(opt, start_factor=0.01, total_iters=warm)
            return sl.SequentialLR(opt, [wu, cos], milestones=[warm]), False
        return cos, False
    # default: step decay (optionally warmed up)
    if not (lr_decay_every and lr_decay_every > 0):
        if warm:
            return sl.LinearLR(opt, start_factor=0.01, total_iters=warm), False
        return None, False
    step = sl.StepLR(opt, step_size=int(lr_decay_every), gamma=float(lr_decay_rate))
    if warm:
        wu = sl.LinearLR(opt, start_factor=0.01, total_iters=warm)
        return sl.SequentialLR(opt, [wu, step], milestones=[warm]), False
    return step, False


def _init_training(backbone, config, resume_from, device, lr,
                   lr_decay_every, lr_decay_rate, seed,
                   train_config=None, resume_strict=False, history_path=None,
                   weight_decay=0.0, lr_schedule="step", warmup_epochs=0,
                   total_epochs=None):
    """Common setup for every train loop: seed the RNGs, build the model and
    optimizer/scheduler -- or restore all of them from `resume_from` (a path or
    a pre-loaded checkpoint dict) so training continues exactly where it
    stopped. Returns (model, meta, opt, sched, start_epoch, best_f1, history,
    sched_is_metric_driven).
    """
    import random
    torch = _require_torch()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    state = None
    if resume_from:
        state = (load_checkpoint(resume_from, device=device)
                 if isinstance(resume_from, str) else resume_from)
        ck_backbone = state.get("backbone") or state["meta"].get("backbone",
                                                                 "diffusionnet")
        if ck_backbone != backbone:
            raise ValueError(f"checkpoint is for backbone {ck_backbone!r}, "
                             f"but --backbone {backbone!r} was requested")
        if train_config is not None:
            _check_resume_config(state.get("train_config"), train_config, resume_strict)
        # rebuild the EXACT architecture from the checkpoint meta, not from
        # the (possibly different) fresh config
        model, _ = build_regressor(backbone, _meta_to_config(backbone, state["meta"]))
        meta = state["meta"]
    else:
        model, meta = build_regressor(backbone, config)

    model = model.to(device)
    if state is not None:
        model.load_state_dict(state["state_dict"])
    # AdamW so --weight-decay is real L2 regularisation (decoupled); wd=0 == Adam.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    sched, sched_metric = _build_scheduler(opt, lr_schedule, lr_decay_every,
                                           lr_decay_rate, warmup_epochs, total_epochs)

    start_epoch, best_f1, history = 0, -1.0, []
    if state is not None:
        if state.get("optimizer"):
            opt.load_state_dict(state["optimizer"])     # state tensors are
        if sched is not None and state.get("scheduler"):  # remapped to the
            sched.load_state_dict(state["scheduler"])     # params' device
        start_epoch = int(state.get("epoch", -1)) + 1
        best_f1 = float(state.get("best_f1", -1.0))
        # prefer the FULL history from the sibling JSON (the checkpoint only
        # embeds a capped tail); fall back to the embedded copy.
        history = _load_history_json(history_path) or list(state.get("history") or [])
        _restore_rng(state.get("rng"))
        logger.info("resumed %s checkpoint (next epoch %d, best %s=%.4f)",
                    backbone, start_epoch,
                    (train_config or {}).get("best_metric", "val F1"), best_f1)
    return model, meta, opt, sched, start_epoch, best_f1, history, sched_metric


def _load_history_json(history_path):
    """Read the full per-eval history list from the sibling JSON, or None."""
    if not history_path:
        return None
    import os
    if not os.path.exists(history_path):
        return None
    try:
        import json as _json
        with open(history_path, encoding="utf-8") as fh:
            data = _json.load(fh)
        return data if isinstance(data, list) else None
    except Exception:                                        # noqa: BLE001
        return None


class _Bookkeeper:
    """Per-epoch bookkeeping shared by all backbones: log the train loss, run
    validation metrics every `eval_every` epochs (accuracy / precision /
    recall / F1 / localisation / angular error), keep the best-by-val-F1 full
    checkpoint at `best_path`, roll a resumable `last` checkpoint at
    `last_path` every `save_every` epochs, persist the metric history to
    `history_path` (JSON), and early-stop after `patience` stale evals."""

    def __init__(self, model, meta, backbone, opt, sched, epochs, *,
                 metrics_fn=None, eval_every=5, patience=0,
                 best_path=None, last_path=None, save_every=1,
                 history_path=None, log_every=1, best_f1=-1.0, history=None,
                 decode=None, train_config=None, best_metric="micro_f1",
                 snapshot_every=0, snapshot_stem=None, sched_metric=False):
        self.model, self.meta, self.backbone = model, meta, backbone
        self.opt, self.sched, self.epochs = opt, sched, epochs
        # a ReduceLROnPlateau scheduler is stepped HERE with the val metric (the
        # train loop must NOT step it per-epoch); others step per-epoch in the loop
        self.sched_metric = bool(sched_metric)
        self.metrics_fn, self.eval_every = metrics_fn, max(1, int(eval_every))
        self.patience = int(patience)
        self.best_path, self.last_path = best_path, last_path
        self.save_every = int(save_every)
        self.history_path, self.log_every = history_path, log_every
        self.best_f1 = float(best_f1)
        self.history = history if history is not None else []
        # decode + train_config are bundled into every saved checkpoint so a raw
        # .ckpt deploys with the run's real operating point (not DEFAULT_DECODE)
        # and a --resume can detect a mismatched continuation.
        self.decode = decode
        self.train_config = train_config
        self.best_metric = best_metric          # micro_f1 (deploy metric) by default
        self.snapshot_every = int(snapshot_every)
        # default snapshot name stem = best_path without its extension
        if snapshot_stem is None and best_path:
            import os
            snapshot_stem = os.path.splitext(best_path)[0]
        self.snapshot_stem = snapshot_stem
        self.stale = 0
        self.saved_best = False
        self.had_valid_eval = False
        self.last_epoch_done = -1

    def _save(self, path, epoch, quiet=True):
        save_checkpoint(path, self.model, self.meta, self.backbone,
                        optimizer=self.opt, scheduler=self.sched, epoch=epoch,
                        best_f1=self.best_f1, history=self.history,
                        decode=self.decode, train_config=self.train_config,
                        best_metric=self.best_metric, quiet=quiet)

    def flush_history(self):
        if not self.history_path:
            return
        import json as _json
        import math
        import os
        d = os.path.dirname(os.path.abspath(self.history_path))
        if d:
            os.makedirs(d, exist_ok=True)
        # NaN (e.g. loc/ang error with zero matches) is not valid strict JSON
        clean = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in rec.items()} for rec in self.history]
        with open(self.history_path, "w", encoding="utf-8") as fh:
            _json.dump(clean, fh, indent=2)

    def after_epoch(self, ep, mean_loss):
        """End-of-epoch hook. Returns True when training should stop early."""
        self.last_epoch_done = ep
        if self.log_every and (ep % self.log_every == 0 or ep == self.epochs - 1):
            logger.info("epoch %3d/%d  train loss=%.5f", ep, self.epochs, mean_loss)

        stop = False
        if self.metrics_fn is not None and (ep % self.eval_every == 0
                                            or ep == self.epochs - 1):
            logger.info("  epoch %d: running validation eval "
                        "(slow on large parts) ...", ep)
            m = self.metrics_fn(self.model, self.meta) or {}
            rec = {"epoch": ep, "train_loss": float(mean_loss)}
            rec.update({"val_" + k: v for k, v in m.items()})
            self.history.append(rec)
            nan = float("nan")
            # select the BEST checkpoint on the same metric we report/deploy
            # (micro_f1 by default), not the per-part macro mean.
            sel = m.get(self.best_metric, nan)
            pct = lambda v: v * 100.0 if isinstance(v, float) else nan
            logger.info("  epoch %d  VAL  accuracy=%.1f%%  precision=%.1f%%  "
                        "recall=%.1f%%  F1=%.1f%% (micro %.1f%%)  loc=%.2f mm  ang=%.1f deg",
                        ep, pct(m.get("accuracy", nan)), pct(m.get("precision", nan)),
                        pct(m.get("recall", nan)), pct(m.get("f1", nan)),
                        pct(m.get("micro_f1", nan)),
                        m.get("mean_loc_err_mm", nan), m.get("mean_ang_err_deg", nan))
            if isinstance(sel, float) and not np.isnan(sel):
                self.had_valid_eval = True
                # metric-driven LR schedule (ReduceLROnPlateau) steps on the val metric
                if self.sched_metric and self.sched is not None:
                    self.sched.step(sel)
                if sel > self.best_f1:
                    self.best_f1, self.stale = float(sel), 0
                    if self.best_path:
                        self._save(self.best_path, ep)
                        self.saved_best = True
                        logger.info("  new best %s=%.4f -> %s", self.best_metric,
                                    sel, self.best_path)
                else:
                    self.stale += 1
                    logger.info("  no %s improvement for %d eval(s) (best=%.4f)",
                                self.best_metric, self.stale, self.best_f1)
                stop = bool(self.patience) and self.stale >= self.patience
            self.flush_history()

        # periodic, never-overwritten snapshots so an earlier good model is
        # recoverable if a later "best" turns out to overfit.
        if (self.snapshot_every and self.snapshot_stem
                and (ep + 1) % self.snapshot_every == 0):
            snap = "%s_ep%04d.ckpt" % (self.snapshot_stem, ep)
            self._save(snap, ep, quiet=False)
            logger.info("  snapshot -> %s", snap)

        if self.last_path and ((self.save_every > 0 and (ep + 1) % self.save_every == 0)
                               or ep == self.epochs - 1 or stop):
            self._save(self.last_path, ep)
            # visible confirmation every 10 epochs (and at the end / early stop) so
            # you know there's a fresh resumable checkpoint and it's safe to stop.
            if (ep + 1) % 10 == 0 or ep == self.epochs - 1 or stop:
                logger.info("  checkpoint saved at epoch %d -> %s  "
                            "[safe to stop; resume with --resume]", ep, self.last_path)
        return stop

    def finalize(self):
        """After the loop: make sure a best checkpoint exists even without a
        validation set (no metrics_fn) and the history file is written. When
        valid evals DID happen but none beat the resumed best_f1, the existing
        best checkpoint is deliberately left untouched -- the final model is
        worse than the historical best."""
        if (self.best_path and not self.saved_best and not self.had_valid_eval
                and self.last_epoch_done >= 0):
            # no eval ever produced a valid metric (no val set, or every eval was
            # NaN because TP+FP+FN=0) -- still leave a usable best from the final
            # epoch so deployment is never left without an artifact.
            if self.metrics_fn is not None:
                logger.warning("no valid validation metric was ever produced "
                               "(all NaN?) -- saving the final-epoch model as best")
            self._save(self.best_path, self.last_epoch_done)
        self.flush_history()


# ---------------------------------------------------------------------------
# loss: heatmap BCE + masked offset L1 + masked direction cosine
# ---------------------------------------------------------------------------

def cp_loss(pred, target, mask, w_heat=1.0, w_off=5.0, w_dir=1.0,
            heat_pos_weight=50.0, heat_loss="bce", focal_gamma=2.0,
            offset_heat_weight=True, part_weight=1.0):
    """Combined regression loss.

    pred   : (N,7) raw model output (heatmap channel is a logit).
    target : (N,7) from prepare_sample (heatmap in [0,1]; offset in NORMALISED
             units, i.e. mm/scale -- see prepare_sample). Because the offset is
             already scale-normalised, no per-mesh rescaling happens here.
    mask   : (N,) bool, vertices near a CP (offset/direction supervised there).
    offset_heat_weight : weight each masked vertex's offset/direction error by its
             TARGET heat, so the vertices that will actually VOTE at decode (high
             heat) are the ones whose offsets are pushed most accurate -- couples
             the otherwise-independent heat and offset heads.
    part_weight : scalar multiplier on this part's total loss (e.g. #keypoints) so
             a multi-CP part is not down-weighted to the same step as a 1-CP part.
    heat_loss : "bce"   -> per-vertex BCE weighted by (1 + heat_pos_weight*h) so
                           the few high-heat vertices are not drowned out;
                "focal" -> quality-focal loss |h - sigmoid(logit)|^gamma * BCE,
                           which down-weights the easy ~0 background and tends to
                           improve precision on the sparse-keypoint heatmap.
                "centernet" -> penalty-reduced focal loss (CornerNet/CenterNet,
                           Law&Deng 2018 / Zhou 2019) NORMALISED BY #KEYPOINTS, not
                           by #vertices. The heat target is ~99% background, so a
                           per-vertex mean() (bce/focal) dilutes the handful of
                           positive vertices into the noise -- the model then either
                           collapses to ~0 (focal/low pos_weight) or over-fires
                           broadly (high pos_weight). Dividing the loss by the number
                           of positive (peak) vertices removes that dilution, which
                           is what lets the heatmap learn sharp, selective peaks.
    """
    torch = _require_torch()
    import torch.nn.functional as Fnn
    heat_logit = pred[:, ct.HEATMAP]
    heat_tgt = target[:, ct.HEATMAP]
    if heat_loss == "centernet":
        # alpha focuses on hard examples; beta reduces the penalty on the Gaussian
        # skirt around each peak (near-positive vertices are not punished as hard
        # negatives). Positives are the peak vertices (target near 1).
        alpha, beta = 2.0, 4.0
        p = torch.sigmoid(heat_logit).clamp(1e-6, 1.0 - 1e-6)
        pos = (heat_tgt >= 1.0 - 1e-4).float()   # exact peaks (encode_targets snaps them)
        neg = 1.0 - pos
        neg_w = (1.0 - heat_tgt).pow(beta)
        pos_loss = ((1.0 - p).pow(alpha) * torch.log(p)) * pos
        neg_loss = (p.pow(alpha) * torch.log(1.0 - p)) * neg_w * neg
        n_pos = pos.sum().clamp(min=1.0)
        loss_heat = -(pos_loss.sum() + neg_loss.sum()) / n_pos
    elif heat_loss == "focal":
        bce = Fnn.binary_cross_entropy_with_logits(heat_logit, heat_tgt,
                                                   reduction="none")
        mod = (heat_tgt - torch.sigmoid(heat_logit)).abs().pow(focal_gamma)
        loss_heat = (mod * bce).mean()
    else:
        heat_w = 1.0 + heat_pos_weight * heat_tgt
        loss_heat = Fnn.binary_cross_entropy_with_logits(
            heat_logit, heat_tgt, weight=heat_w)

    if mask.any():
        m = mask
        off_err = (pred[m][:, ct.OFFSET] - target[m][:, ct.OFFSET]).abs().mean(dim=-1)
        dir_pred = Fnn.normalize(pred[m][:, ct.DIRECTION], dim=-1, eps=1e-8)
        dir_tgt = Fnn.normalize(target[m][:, ct.DIRECTION], dim=-1, eps=1e-8)
        dir_err = 1.0 - (dir_pred * dir_tgt).sum(-1)
        if offset_heat_weight:
            w = heat_tgt[m].clamp(min=1e-3)          # vote weight = target heat
            wsum = w.sum().clamp(min=1e-6)
            loss_off = (w * off_err).sum() / wsum
            loss_dir = (w * dir_err).sum() / wsum
        else:
            loss_off = off_err.mean()
            loss_dir = dir_err.mean()
    else:
        loss_off = pred.sum() * 0.0
        loss_dir = pred.sum() * 0.0

    total = (w_heat * loss_heat + w_off * loss_off + w_dir * loss_dir) * part_weight
    n_pos = float((heat_tgt >= 1.0 - 1e-4).sum())
    return total, {"heat": float(loss_heat.detach()), "off": float(loss_off.detach()),
                   "dir": float(loss_dir.detach()), "total": float(total.detach()),
                   "n_pos": n_pos}


def pred_to_array(pred, offset_scale=1.0):
    """Convert raw model output -> (N,7) numpy ready for cp_targets.decode_predictions
    (sigmoid on heatmap, unit-normalise direction).

    offset_scale : the model predicts offsets in NORMALISED units; multiply by the
    mesh scale (mm) here so decode_predictions, which adds the offset to the raw
    mm vertices, sees millimetres. Pass the sample's 'scale'.
    """
    torch = _require_torch()
    import torch.nn.functional as Fnn
    out = pred.detach().clone()
    out[:, ct.HEATMAP] = torch.sigmoid(out[:, ct.HEATMAP])
    out[:, ct.OFFSET] = out[:, ct.OFFSET] * float(offset_scale)
    out[:, ct.DIRECTION] = Fnn.normalize(out[:, ct.DIRECTION], dim=-1, eps=1e-8)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# MLP training loop (smoke-test path). Samples: dicts with verts, target, mask, scale.
# ---------------------------------------------------------------------------

def train_cpmlp(samples, epochs=300, lr=1e-3, width=128, depth=4,
                device="cpu", log_every=50, metrics_fn=None, eval_every=10,
                patience=0, best_path=None, last_path=None, save_every=1,
                history_path=None, resume_from=None, seed=0,
                decode=None, train_config=None, best_metric="micro_f1",
                resume_strict=False, snapshot_every=0, weight_decay=0.0,
                offset_heat_weight=True, part_weight_mode="none", grad_clip=0.0,
                lr_schedule="step", warmup_epochs=0):
    """Overfit/smoke train the MLP backbone on prepared samples.

    samples: list of {'verts_norm' (N,3), 'target' (N,7), 'mask' (N,), 'scale'}.
    Supports the same per-epoch val metrics + best/last full-state .ckpt +
    resume machinery as the graph backbones (see train_diffusionnet_regressor
    for the parameter docs). Returns the trained model.
    """
    torch = _require_torch()
    model, meta, opt, _, start_epoch, best_f1, history, _sm = _init_training(
        "mlp", {"width": width, "depth": depth}, resume_from, device, lr,
        0, 0.5, seed, train_config=train_config, resume_strict=resume_strict,
        history_path=history_path, weight_decay=weight_decay)
    if start_epoch >= epochs:
        logger.info("checkpoint already at epoch %d >= --epochs %d; nothing to "
                    "train (raise --epochs to continue)", start_epoch, epochs)
        return model
    tensors = []
    for s in samples:
        tensors.append({
            "x": torch.tensor(s["verts_norm"], dtype=torch.float32, device=device),
            "t": torch.tensor(s["target"], dtype=torch.float32, device=device),
            "m": torch.tensor(s["mask"], dtype=torch.bool, device=device),
            "scale": float(s["scale"]),
        })
    bk = _Bookkeeper(model, meta, "mlp", opt, None, epochs,
                     metrics_fn=metrics_fn, eval_every=eval_every,
                     patience=patience, best_path=best_path, last_path=last_path,
                     save_every=save_every, history_path=history_path,
                     log_every=log_every, best_f1=best_f1, history=history,
                     decode=decode, train_config=train_config,
                     best_metric=best_metric, snapshot_every=snapshot_every)
    try:
        for ep in range(start_epoch, epochs):
            model.train()
            tot = 0.0
            for s in tensors:
                opt.zero_grad(set_to_none=True)
                out = model(s["x"])
                pw = (max(1.0, float((s["t"][:, ct.HEATMAP] >= 1.0 - 1e-4).sum()))
                      if part_weight_mode == "keypoints" else 1.0)
                loss, parts = cp_loss(out, s["t"], s["m"],
                                      offset_heat_weight=offset_heat_weight,
                                      part_weight=pw)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                tot += parts["total"]
            if bk.after_epoch(ep, tot / max(1, len(tensors))):
                logger.info("early stopping at epoch %d", ep)
                break
    except KeyboardInterrupt:
        bk.flush_history()
        logger.warning("training interrupted -- resume from the last checkpoint%s",
                       f" ({last_path})" if last_path else "")
        raise
    bk.finalize()
    return model


def prepare_sample(part, dedup=True, cp_surface_tol_frac=None, quiet=False):
    """Part -> training sample dict (normalised verts, encoded target, mask, scale).

    Uses deduped terminal-block locations as the regression targets. Carries the
    raw verts/faces too so the DiffusionNet backbone can build mesh operators.

    cp_surface_tol_frac: if set, DROP any CP whose nearest vertex is farther than
    this fraction of the bbox diagonal (a CP floating off the surface is likely a
    mislabel). When None, off-surface CPs are only reported and kept.
    quiet: suppress the per-part validation / dedup / sigma-cap log lines. Set for
    AUGMENTED clones -- a clone is a rigid (distance-preserving) copy of its base,
    so its warnings are IDENTICAL to the original's and re-printing them once per
    clone just floods the terminal (and made augmentation look like the cause of
    off-surface CPs, which it provably is not -- rotation is an isometry).
    """
    import logging as _logging
    _prev_disable = _logging.root.manager.disable
    if quiet:                       # silence redundant clone logs (dedup/sigma/surface)
        _logging.disable(_logging.WARNING)
    try:
        return _prepare_sample_impl(part, dedup, cp_surface_tol_frac)
    finally:
        if quiet:
            _logging.disable(_prev_disable)


def _prepare_sample_impl(part, dedup, cp_surface_tol_frac):
    import json_dataset as jd
    if dedup:
        _, bp, bd = jd.dedup_connection_points(part)
    else:
        bp = np.asarray(part.cp_points, dtype=np.float64)
        bd = np.asarray(part.cp_directions, dtype=np.float64)

    V = np.asarray(part.vertices, dtype=np.float64)
    # CP-on-surface check. Most off-surface CPs are RECESSED terminals (a contact a
    # few mm inside a socket) -- legitimate and now learnable (encode_targets forces
    # the peak vertex into the offset mask). Only a CP very far from the mesh is a
    # likely real mislabel (wrong frame/units), so tier the two instead of crying
    # "mislabel" at every recessed terminal.
    if len(bp) and len(V):
        diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
        sdist = ct.cp_surface_distances(V, bp)
        recessed = (sdist > 0.05 * diag) & (sdist <= 0.15 * diag)
        mislabel = sdist > 0.15 * diag
        if recessed.any():
            logger.info("part %s: %d CP(s) recessed 5-15%% of bbox from the surface "
                        "(expected for socketed terminals; kept)",
                        part.part_nr, int(recessed.sum()))
        if mislabel.any():
            logger.warning("part %s: %d/%d CP(s) lie >15%% of bbox from any vertex "
                           "(max %.1fmm) -- LIKELY MISLABEL (wrong frame/units). Drop "
                           "with --cp-surface-tol-frac 0.15", part.part_nr,
                           int(mislabel.sum()), len(bp), float(sdist.max()))
        if cp_surface_tol_frac is not None:
            keep = sdist <= cp_surface_tol_frac * diag
            if not keep.all():
                logger.warning("part %s: dropping %d off-surface CP(s) (> %.1f%% bbox)",
                               part.part_nr, int((~keep).sum()),
                               100.0 * cp_surface_tol_frac)
                bp, bd = bp[keep], bd[keep]

    Vn, center, scale = normalize_vertices(part.vertices)
    target, mask, sigma = ct.encode_targets(part.vertices, bp, bd)
    # #3: regress offsets in NORMALISED units (mm/scale) so the target is
    # scale-invariant and matches the scale-normalised input features. The mm
    # offset is recovered at decode via pred_to_array(offset_scale=scale).
    target = target.copy()
    target[:, ct.OFFSET] = target[:, ct.OFFSET] / scale
    return {"part_nr": str(part.part_nr),
            "verts_norm": Vn, "verts": np.asarray(part.vertices, dtype=np.float64),
            "faces": np.asarray(part.faces, dtype=np.int64),
            "target": target, "mask": mask, "scale": scale,
            "center": center, "sigma": sigma, "gt_points": bp, "gt_directions": bd}


def augment_part(part, rng, rotate=True, jitter_frac=0.0, reflect=False):
    """Return a randomly augmented clone of a Part for training.

    A *rigid* rotation (uniform random, about the vertex centroid) is applied to
    the whole part at once -- vertices, CP points AND CP directions -- so the
    geometry/label relationship is preserved exactly. This is the lever that
    teaches the raw-xyz backbones (mlp/knngraph, which have NO built-in pose
    invariance) to detect connection points at any orientation; without it they
    overfit to the single canonical pose each part ships in.

    reflect (>0 prob): with probability 0.5, also mirror the part across a random
    axis through the centroid. Connector geometry is frequently mirror-symmetric,
    so reflections are plausible unseen parts. A reflection is improper (det=-1),
    so vertices AND CP points/directions are all mirrored together, and face
    winding is flipped to keep outward normals consistent for the mesh backbones.

    jitter_frac (>0) adds Gaussian vertex noise with std = jitter_frac * bbox
    diagonal to the VERTICES ONLY -- the GT CP points stay put, so the network
    learns to localise the true connection point from a noisy/imperfect mesh
    (encode_targets is recomputed downstream against the jittered vertices).

    terminal names are unchanged. Used to expand the TRAIN set only; the
    validation set is never augmented (so val metrics stay deployment-real).
    """
    import json_dataset as jd
    import augment as aug
    V = np.asarray(part.vertices, dtype=np.float64)
    cp = np.asarray(part.cp_points, dtype=np.float64)
    cd = np.asarray(part.cp_directions, dtype=np.float64)
    F = np.asarray(part.faces, dtype=np.int64)
    if rotate and len(V):
        R = aug._random_rotation_matrix(rng)
        c = V.mean(0)
        V = (V - c) @ R.T + c
        if len(cp):
            cp = (cp - c) @ R.T + c
            cd = cd @ R.T                      # directions rotate, stay unit
    if reflect and len(V) and rng.random() < 0.5:
        ax = int(rng.integers(3))             # mirror across one random axis plane
        c = V.mean(0)
        V = V.copy(); V[:, ax] = 2.0 * c[ax] - V[:, ax]
        if len(cp):
            cp = cp.copy(); cp[:, ax] = 2.0 * c[ax] - cp[:, ax]
            cd = cd.copy(); cd[:, ax] = -cd[:, ax]   # direction mirrors too
        if F.size:                            # reflection flips winding -> swap 2 cols
            F = F[:, [0, 2, 1]]
    if jitter_frac and jitter_frac > 0.0 and len(V):
        diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
        V = V + rng.normal(0.0, jitter_frac * diag, size=V.shape)
    return jd.Part(part_nr=str(part.part_nr) + "~aug", vertices=V,
                   faces=F, cp_points=cp, cp_directions=cd,
                   cp_names=list(part.cp_names))


# ---------------------------------------------------------------------------
# production backbone: DiffusionNet training / inference (needs diffusion_net)
# ---------------------------------------------------------------------------
# The eigenbasis (mesh operators) is the expensive step at 479-file scale; it is
# computed ONCE per part and cached to op_cache_dir (diffusion_net hashes the
# vertices to key the cache), then reused across epochs and runs. Operators are
# built on the raw mm mesh; the *input features* are the normalised xyz so the
# first layer is well-conditioned regardless of part size. Offset targets are
# scale-normalised (mm/scale) to match the normalised input; pred_to_array
# multiplies them back by the mesh scale before decoding.

def _cp_operators(verts, faces, k_eig=128, op_cache_dir=None):
    """Mesh operators (mass/Laplacian/eigenbasis/gradients) for one part.

    Cached to op_cache_dir when given (recommended for the 479-file corpus).
    """
    torch = _require_torch()
    import diffusionnet as dnmod
    dn = dnmod._require(
        "diffusion_net",
        "clone https://github.com/nmwsharp/diffusion-net and add its src/ to "
        "PYTHONPATH; needs robust_laplacian + potpourri3d + scikit-learn")
    k_eig = int(min(int(k_eig), dnmod.MAX_K_EIG))
    V = torch.tensor(np.asarray(verts), dtype=torch.float32)
    F = torch.tensor(np.asarray(faces), dtype=torch.long)
    _, mass, L, evals, evecs, gradX, gradY = dn.geometry.get_operators(
        V, F, k_eig=k_eig, op_cache_dir=op_cache_dir)
    return {"verts": V, "faces": F, "mass": mass, "L": L, "evals": evals,
            "evecs": evecs, "gradX": gradX, "gradY": gradY}


def _to_device_ops(ops, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ops.items()}


def train_diffusionnet_regressor(train_samples, config=None, epochs=200, lr=1e-3,
                                 device="cpu", op_cache_dir=None, log_every=1,
                                 metrics_fn=None, eval_every=5, patience=0,
                                 w_heat=1.0, w_off=5.0, w_dir=1.0,
                                 heat_pos_weight=50.0, heat_loss="bce",
                                 focal_gamma=2.0, max_gpu_verts=60000,
                                 best_path=None, last_path=None, save_every=1,
                                 history_path=None, resume_from=None, seed=0,
                                 lr_decay_every=0, lr_decay_rate=0.5,
                                 accum_steps=1, low_memory=False,
                                 decode=None, train_config=None,
                                 best_metric="micro_f1", resume_strict=False,
                                 snapshot_every=0, weight_decay=0.0,
                                 lr_schedule="step", warmup_epochs=0,
                                 offset_heat_weight=True, part_weight_mode="none",
                                 grad_clip=0.0):
    """Train the production DiffusionNet regressor (C_out=7).

    Each sample needs 'verts','faces','verts_norm','target','mask','scale'
    (see prepare_sample). Operators are precomputed/cached once per part, each
    part is moved to the device for its step then freed. Returns (model, meta).

    seed                 : seeds torch/np/random for reproducible runs (#4).
    lr_decay_every/rate  : StepLR schedule (epochs / gamma); 0 disables it (#4).
    accum_steps          : gradient accumulation over N parts before opt.step.
    metrics_fn(model,meta): optional; runs every `eval_every` epochs and returns
                           the val metric dict (accuracy/precision/recall/f1/
                           loc/ang). Drives best-by-F1 checkpointing to
                           `best_path` and early stopping after `patience`
                           stale evals.
    best_path/last_path  : full-state .ckpt files -- best-by-val-F1 and a rolling
                           resumable snapshot (every `save_every` epochs and at
                           the end). Both contain model+optimizer+scheduler+
                           epoch+RNG+history, so EITHER can be resumed from.
    history_path         : per-eval metric history as JSON (plot/inspect later).
    resume_from          : path (or loaded dict) of a previous .ckpt; training
                           continues at its next epoch toward `epochs` total --
                           works across machines (commit the .ckpt, pull, resume).
    max_gpu_verts        : parts above this run on CPU (4 GB GPU OOMs on big
                           meshes); grads from the temp CPU copy are added back to
                           the GPU optimiser, identical to a GPU step.
    low_memory           : do NOT hold every part's eigenbasis resident; reload it
                           from op_cache_dir each step (bounded RAM for corpora
                           larger than memory, at the cost of cache I/O) (#7).
    """
    import copy
    torch = _require_torch()
    import diffusionnet as dnmod
    model, meta, opt, sched, start_epoch, best_f1, history, sched_metric = _init_training(
        "diffusionnet", config, resume_from, device, lr,
        lr_decay_every, lr_decay_rate, seed, train_config=train_config,
        resume_strict=resume_strict, history_path=history_path,
        weight_decay=weight_decay, lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs, total_epochs=epochs)
    if start_epoch >= epochs:
        logger.info("checkpoint already at epoch %d >= --epochs %d; nothing to "
                    "train (raise --epochs to continue)", start_epoch, epochs)
        return model, meta
    accum = max(1, int(accum_steps))

    def _loss(out, t, m):
        pw = 1.0
        if part_weight_mode == "keypoints":
            pw = max(1.0, float((t[:, ct.HEATMAP] >= 1.0 - 1e-4).sum()))
        return cp_loss(out, t, m, w_heat=w_heat, w_off=w_off, w_dir=w_dir,
                       heat_pos_weight=heat_pos_weight, heat_loss=heat_loss,
                       focal_gamma=focal_gamma, offset_heat_weight=offset_heat_weight,
                       part_weight=pw)

    def _accumulate(ops, d, run_device, src_device, loss_div):
        """Forward+backward for one part, accumulating gradient into `model`
        (no zero_grad / no step here -- the caller steps at accumulation
        boundaries). Oversized/OOM parts run on a temporary copy on run_device
        and copy their gradient back to the model on src_device."""
        ops_d = _to_device_ops(ops, run_device)
        x = (d["x"].to(run_device) if meta["input_features"] == "xyz"
             else dnmod._model_input(ops_d, meta))
        t = d["t"].to(run_device); m = d["m"].to(run_device)
        if run_device == src_device:
            out = dnmod._forward(model, ops_d, x)
            loss, parts = _loss(out, t, m)
            (loss / loss_div).backward()
        else:
            cm = copy.deepcopy(model).to(run_device)
            out = dnmod._forward(cm, ops_d, x)
            loss, parts = _loss(out, t, m)
            (loss / loss_div).backward()
            for pg, pc in zip(model.parameters(), cm.parameters()):
                if pc.grad is not None:
                    g = pc.grad.detach().to(src_device)
                    pg.grad = g if pg.grad is None else (pg.grad + g)
            del cm
        del ops_d, x, t, m, out, loss
        return parts

    # #1: precompute operators with a per-part guard -- one malformed mesh
    # (degenerate faces, eigensolver non-convergence) must not kill the whole run.
    logger.info("DiffusionNet regressor: preparing operators for %d parts "
                "(k_eig=%d, cache=%s, low_memory=%s) ...", len(train_samples),
                meta["k_eig"], op_cache_dir, low_memory)
    prepared, skipped = [], 0
    for s in train_samples:
        item = {"x": torch.tensor(s["verts_norm"], dtype=torch.float32),
                "t": torch.tensor(s["target"], dtype=torch.float32),
                "m": torch.tensor(s["mask"], dtype=torch.bool),
                "scale": float(s["scale"]),
                "verts": s["verts"], "faces": s["faces"]}
        try:
            ops = _cp_operators(s["verts"], s["faces"], meta["k_eig"], op_cache_dir)
        except Exception as exc:                      # noqa: BLE001 (skip bad mesh)
            skipped += 1
            logger.warning("skipping part (operator build failed, %d verts): %s",
                           len(np.asarray(s["verts"])), exc)
            continue
        if not low_memory:
            item["ops"] = ops                          # keep resident
        del ops                                        # low_memory: only warmed cache
        prepared.append(item)
    if skipped:
        logger.warning("skipped %d/%d parts with bad geometry", skipped,
                       len(train_samples))
    if not prepared:
        raise RuntimeError("no usable parts after operator precompute")

    import random
    bk = _Bookkeeper(model, meta, "diffusionnet", opt, sched, epochs,
                     metrics_fn=metrics_fn, eval_every=eval_every,
                     patience=patience, best_path=best_path, last_path=last_path,
                     save_every=save_every, history_path=history_path,
                     log_every=log_every, best_f1=best_f1, history=history,
                     decode=decode, train_config=train_config,
                     best_metric=best_metric, snapshot_every=snapshot_every,
                     sched_metric=sched_metric)
    order = list(range(len(prepared)))
    try:
        for ep in range(start_epoch, epochs):
            model.train()
            random.shuffle(order)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for k, i in enumerate(order, 1):
                d = prepared[i]
                ops = (_cp_operators(d["verts"], d["faces"], meta["k_eig"], op_cache_dir)
                       if low_memory else d["ops"])
                n = d["x"].shape[0]
                big = device != "cpu" and max_gpu_verts and n > max_gpu_verts
                try:
                    parts = _accumulate(ops, d, "cpu" if big else device, device, accum)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    logger.warning("CUDA OOM on %d-vertex part -> CPU fallback", n)
                    parts = _accumulate(ops, d, "cpu", device, accum)
                if low_memory:
                    del ops
                if k % accum == 0 or k == len(order):
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                tot += parts["total"]
            if sched is not None and not sched_metric:
                sched.step()
            if bk.after_epoch(ep, tot / max(1, len(prepared))):
                logger.info("early stopping at epoch %d", ep)
                break
    except KeyboardInterrupt:
        bk.flush_history()
        logger.warning("training interrupted -- resume from the last checkpoint%s",
                       f" ({last_path})" if last_path else "")
        raise
    bk.finalize()
    return model, meta


def infer_diffusionnet(model, meta, verts, faces, verts_norm,
                       op_cache_dir=None, device="cpu", max_gpu_verts=60000,
                       offset_scale=1.0):
    """Run a trained DiffusionNet regressor on one part -> (N,7) numpy array
    ready for cp_targets.decode_predictions.

    offset_scale : mesh scale (mm); the model emits normalised offsets, so this
    converts them back to millimetres for the decoder (pass the sample 'scale').
    Like training, oversized meshes (or a CUDA OOM) fall back to the CPU so a
    4 GB GPU does not crash on the largest parts; results are identical.
    """
    torch = _require_torch()
    import diffusionnet as dnmod
    import copy
    n = len(np.asarray(verts))
    run_device = ("cpu" if (device != "cpu" and max_gpu_verts and n > max_gpu_verts)
                  else device)

    def _run(rd):
        ops = _to_device_ops(_cp_operators(verts, faces, meta["k_eig"], op_cache_dir), rd)
        x = (torch.tensor(np.asarray(verts_norm), dtype=torch.float32, device=rd)
             if meta["input_features"] == "xyz" else dnmod._model_input(ops, meta))
        mdl = model if rd == device else copy.deepcopy(model).to(rd)
        mdl.eval()
        with torch.no_grad():
            out = dnmod._forward(mdl, ops, x)
        return pred_to_array(out, offset_scale=offset_scale)

    try:
        return _run(run_device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        logger.warning("CUDA OOM during inference on %d-vertex part -> CPU", n)
        return _run("cpu")


# ===========================================================================
# kNN-graph backbone (native-Windows: NO robust_laplacian / potpourri3d)
# ===========================================================================
# A torch-only geometric backbone for environments where the DiffusionNet native
# wheels segfault (Windows). The "operator" is a k-nearest-neighbour graph built
# with scipy.spatial.cKDTree; message passing is plain torch (EdgeConv: each
# vertex aggregates an MLP of (h_i, h_j - h_i) over its neighbours, max-pooled).
# Stacking blocks with residuals grows a real geometric receptive field, so --
# unlike the bare MLP -- it generalises across parts. It reuses the SAME target
# encoding / cp_loss / decode / metrics as the other backbones; only the network
# and the operator differ. Deps: torch + scipy + numpy (all clean Windows wheels).

KNN_DEFAULTS = {"c_width": 128, "n_layers": 4, "k": 16, "global_feat": False,
                "dropout": 0.0}


def _knn_graph(verts, k=16):
    """(N,k) neighbour-index tensor from a point set, via scipy cKDTree.

    Built on the (uniformly) normalised coords -- uniform scale/translation
    preserves nearest neighbours, so the graph is scale-invariant. k is clamped
    to N-1 for tiny meshes. Needs no mesh faces (works on point clouds too).

    workers=-1 parallelises the neighbour search across all cores -- a big win at
    479-parts x (1 + --augment) graphs; each is built once and cached.
    """
    torch = _require_torch()
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]
    V = np.asarray(verts, dtype=np.float64)
    n = len(V)
    kq = int(min(k + 1, n))                  # +1: query returns the point itself
    tree = cKDTree(V)
    _, idx = tree.query(V, k=kq, workers=-1)
    idx = np.atleast_2d(idx)
    if idx.shape[1] > 1:
        idx = idx[:, 1:]                     # drop self-neighbour (column 0)
    return torch.tensor(np.ascontiguousarray(idx), dtype=torch.long)


def build_knngraph_regressor(config=None):
    """Build the EdgeConv kNN-graph regressor (C_out=7). Returns (model, meta)."""
    _require_torch()
    import torch
    import torch.nn as nn
    cfg = {**KNN_DEFAULTS, **(config or {})}
    width = int(cfg["c_width"]); n_layers = int(cfg["n_layers"]); c_in = 3
    global_feat = bool(cfg["global_feat"])
    dropout = float(cfg.get("dropout", 0.0))

    class EdgeConv(nn.Module):
        """h_i' = max_j MLP([h_i, h_j - h_i]) over the kNN neighbours j of i."""
        def __init__(self, c_in, c_out):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2 * c_in, c_out),
                                     nn.LayerNorm(c_out), nn.ReLU(),
                                     nn.Linear(c_out, c_out),
                                     nn.LayerNorm(c_out))

        def forward(self, h, nbr_idx):
            hj = h[nbr_idx]                              # (N, k, C)
            hi = h.unsqueeze(1).expand_as(hj)           # (N, k, C)
            edge = torch.cat([hi, hj - hi], dim=-1)     # (N, k, 2C)
            return self.mlp(edge).amax(dim=1)           # (N, C') max aggregation

    class KNNGraphNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(c_in, width), nn.ReLU())
            self.blocks = nn.ModuleList([EdgeConv(width, width) for _ in range(n_layers)])
            self.global_feat = global_feat
            # dropout before the head regularises the per-vertex features (the main
            # in-network defence against overfitting on a small corpus)
            drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            if global_feat:
                # PointNet-style global context: each vertex sees a summary of the
                # WHOLE part, which helps disambiguate sparse connection points
                # (a salient bump is a CP only relative to the rest of the housing).
                self.head = nn.Sequential(drop, nn.Linear(2 * width, width), nn.ReLU(),
                                          nn.Linear(width, N_CHANNELS))
            else:
                self.head = nn.Sequential(drop, nn.Linear(width, N_CHANNELS))

        def forward(self, x, nbr_idx):
            h = self.inp(x)
            for blk in self.blocks:
                h = h + blk(h, nbr_idx)                  # residual -> stable depth
            if self.global_feat:
                g = h.amax(dim=0, keepdim=True).expand_as(h)   # global max-pool
                h = torch.cat([h, g], dim=-1)
            return self.head(h)

    meta = {"backbone": "knngraph", "input_features": "xyz", "c_in": c_in,
            "c_width": width, "n_layers": n_layers, "k": int(cfg["k"]),
            "global_feat": global_feat, "dropout": dropout}
    return KNNGraphNet(), meta


def save_knngraph(model, meta, path):
    """Save a trained kNN-graph regressor (weights + meta) to `path`."""
    save_checkpoint(path, model, meta, "knngraph")


def load_knngraph(path, device="cpu"):
    """Rebuild a kNN-graph regressor from a checkpoint. Returns (model, meta)."""
    model, meta, _ = load_model(path, device=device)
    return model, meta


def train_knngraph_regressor(train_samples, config=None, epochs=200, lr=1e-3,
                             device="cpu", log_every=1, metrics_fn=None,
                             eval_every=5, patience=0,
                             w_heat=1.0, w_off=5.0, w_dir=1.0, heat_pos_weight=50.0,
                             heat_loss="bce", focal_gamma=2.0, max_gpu_verts=60000,
                             best_path=None, last_path=None, save_every=1,
                             history_path=None, resume_from=None, seed=0,
                             lr_decay_every=0, lr_decay_rate=0.5, accum_steps=1,
                             amp=False, decode=None, train_config=None,
                             best_metric="micro_f1", resume_strict=False,
                             snapshot_every=0, weight_decay=0.0,
                             lr_schedule="step", warmup_epochs=0,
                             offset_heat_weight=True, part_weight_mode="none",
                             grad_clip=0.0):
    """Train the kNN-graph regressor. Mirrors train_diffusionnet_regressor (seed,
    LR schedule, gradient accumulation, GPU/CPU-hybrid for big meshes, per-epoch
    val metrics, best/last full-state .ckpt + resume, early stop) but the per-part
    operator is a cheap kNN graph held in RAM (no eigenbasis, no op cache).
    Returns (model, meta). See train_diffusionnet_regressor for the checkpoint/
    resume parameter docs.

    weight_decay : AdamW L2 regularisation. lr_schedule : step/plateau/cosine.
    offset_heat_weight : couple offset loss to target heat. part_weight_mode :
    'none' or 'keypoints' (weight a part's loss by its #CPs). grad_clip : max grad
    norm before opt.step (0 = off)."""
    import copy
    torch = _require_torch()
    model, meta, opt, sched, start_epoch, best_f1, history, sched_metric = _init_training(
        "knngraph", config, resume_from, device, lr,
        lr_decay_every, lr_decay_rate, seed, train_config=train_config,
        resume_strict=resume_strict, history_path=history_path,
        weight_decay=weight_decay, lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs, total_epochs=epochs)
    if start_epoch >= epochs:
        logger.info("checkpoint already at epoch %d >= --epochs %d; nothing to "
                    "train (raise --epochs to continue)", start_epoch, epochs)
        return model, meta
    accum = max(1, int(accum_steps))
    # Record the subsample cap as a model property: inference must build the kNN
    # graph at the SAME vertex density it was trained on, so infer_knngraph reads
    # this from meta rather than a (possibly different) CLI value. 0 = no cap.
    meta["subsample"] = int(max_gpu_verts or 0)

    # Mixed precision (fp16 autocast + GradScaler), CUDA only, OPT-IN (amp=True).
    # EdgeConv here is gather/bandwidth-bound, not compute-bound, so on a T1200 the
    # fp16 cast + scaler overhead actually makes BELOW-cliff caps (<=~12k verts)
    # SLOWER than fp32; it only claws back ~30% in the deep memory-thrash regime
    # (cap ~20k) and even then can't beat just capping at 8k. Kept for high-cap runs
    # or a stronger GPU. When off, every scaler call is a no-op so the fp32 path is
    # byte-for-byte unchanged. Scaler state is intentionally NOT checkpointed -- it
    # re-warms its loss scale within a few steps after a resume.
    use_amp = bool(amp) and device == "cuda"
    from torch.amp.grad_scaler import GradScaler   # defining module -> pyright-clean
    scaler = GradScaler("cuda", enabled=use_amp)
    if use_amp:
        logger.info("knngraph: mixed precision ON (fp16 autocast + GradScaler)")

    # one persistent CPU replica for the big-part / OOM fallback, re-synced from the
    # live model each use -- avoids rebuilding the whole module (copy.deepcopy) per
    # big part per epoch.
    _cpu = {"m": None}

    def _cpu_replica():
        if _cpu["m"] is None:
            mm, _ = build_regressor("knngraph", _meta_to_config("knngraph", meta))
            _cpu["m"] = mm.to("cpu")
        _cpu["m"].load_state_dict(model.state_dict())
        _cpu["m"].train()
        return _cpu["m"]

    def _loss(out, t, m):
        # offset in NORMALISED mm can be tiny -- compute the loss in fp32 (outside
        # autocast) so AMP fp16 never underflows the offset/dir terms.
        pw = 1.0
        if part_weight_mode == "keypoints":
            pw = max(1.0, float((t[:, ct.HEATMAP] >= 1.0 - 1e-4).sum()))
        return cp_loss(out.float(), t, m, w_heat=w_heat, w_off=w_off, w_dir=w_dir,
                       heat_pos_weight=heat_pos_weight, heat_loss=heat_loss,
                       focal_gamma=focal_gamma, offset_heat_weight=offset_heat_weight,
                       part_weight=pw)

    def _accumulate(d, run_device, src_device, loss_div):
        x = d["x"].to(run_device); nb = d["nbr"].to(run_device)
        t = d["t"].to(run_device); m = d["m"].to(run_device)
        if run_device == src_device:
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(use_amp and run_device == "cuda")):
                out = model(x, nb)
            loss, parts = _loss(out, t, m)      # fp32 loss, autocast only the forward
            scaler.scale(loss / loss_div).backward()
        else:
            # CPU fallback (oversized part or CUDA OOM): always fp32, no autocast.
            cm = _cpu_replica()
            out = cm(x, nb)
            loss, parts = _loss(out, t, m)
            # Scale these CPU grads by the SAME factor the scaler will later unscale
            # on the GPU model, so a mixed GPU/CPU accumulation window stays
            # consistent (get_scale()==1.0 when amp is off -> identical to fp32).
            (scaler.get_scale() * loss / loss_div).backward()
            for pg, pc in zip(model.parameters(), cm.parameters()):
                if pc.grad is not None:
                    g = pc.grad.detach().to(src_device)
                    pg.grad = g if pg.grad is None else (pg.grad + g)
            cm.zero_grad(set_to_none=True)      # replica is reused, clear its grads
        del x, nb, t, m, out, loss
        return parts

    # kNN graphs are built lazily on first access and cached -- training starts
    # immediately, and each graph is built only once (not once per epoch).
    # Large meshes (>max_gpu_verts) are subsampled to that cap. The subsample is
    # UNIFORM (random) so the kNN-graph density matches what infer_knngraph builds
    # on the same part -- inference subsamples big parts to the same cap, so the
    # model never sees one density at train and another at test. Only the exact CP
    # peaks (heat==1, a handful per part) are force-kept: that protects recall /
    # the centernet positives without skewing the otherwise-uniform density.
    logger.info("kNN-graph regressor: %d parts (k=%d, subsample=%d), graphs cached "
                "on first use", len(train_samples), meta["k"], meta["subsample"])
    prepared = []
    for s in train_samples:
        verts = s["verts_norm"]
        tgt   = s["target"]
        msk   = s["mask"]
        n = len(verts)
        if max_gpu_verts and n > max_gpu_verts:
            # UNIFORM subsample, identical strategy to infer_knngraph, so the model
            # trains on the same vertex distribution it sees at inference (which is
            # GT-free and therefore cannot keep CP peaks). PER-PART deterministic
            # seed so it is stable across shuffle/resume order.
            pid = str(s.get("part_nr", id(s)))
            psd = (int(seed) ^ (int(hashlib.sha256(pid.encode()).hexdigest(), 16)
                                & 0x7FFFFFFF))
            # CP locations (normalised) recovered from the full target's peaks,
            # BEFORE subsampling drops them: vertex + offset = its nearest CP.
            heat = tgt[:, ct.HEATMAP]
            peaks = np.where(heat >= 1.0 - 1e-4)[0]
            cp_norm = verts[peaks] + tgt[peaks][:, ct.OFFSET] if len(peaks) else None
            idx = uniform_subsample_idx(n, max_gpu_verts, psd)
            verts = verts[idx].copy(); tgt = tgt[idx].copy(); msk = msk[idx].copy()
            # Re-snap one heat=1 peak per CP onto the nearest KEPT vertex. This does
            # NOT bias the sampled set (still uniform) -- it only restores the
            # centernet positives the uniform draw may have dropped, so a big part
            # still teaches its CPs instead of collapsing to all-background.
            if cp_norm is not None:
                for c in cp_norm:
                    j = int(np.argmin(np.linalg.norm(verts - c, axis=1)))
                    tgt[j, ct.HEATMAP] = 1.0
        prepared.append({"verts_norm": verts,
                         "nbr": None,  # built on first access, then cached
                         "x": torch.tensor(verts, dtype=torch.float32),
                         "t": torch.tensor(tgt,   dtype=torch.float32),
                         "m": torch.tensor(msk,   dtype=torch.bool)})
    if not prepared:
        raise RuntimeError("no usable parts for kNN-graph training")

    import random
    import time
    bk = _Bookkeeper(model, meta, "knngraph", opt, sched, epochs,
                     metrics_fn=metrics_fn, eval_every=eval_every,
                     patience=patience, best_path=best_path, last_path=last_path,
                     save_every=save_every, history_path=history_path,
                     log_every=log_every, best_f1=best_f1, history=history,
                     decode=decode, train_config=train_config,
                     best_metric=best_metric, snapshot_every=snapshot_every,
                     sched_metric=sched_metric)
    order = list(range(len(prepared)))
    # within-epoch heartbeat (every ~20% of parts). The first epoch is the longest
    # silent stretch (every kNN graph is built and big parts first hit the CPU
    # path), but even cached-graph epochs run 1-2 min on this corpus -- silent for
    # that long right after a 0.0% epoch-0 F1 looks exactly like a hang, so the
    # heartbeat fires EVERY epoch, not just the first.
    hb = max(1, len(order) // 5)
    try:
        for ep in range(start_epoch, epochs):
            model.train()
            random.shuffle(order)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            ep_t0 = time.time()
            for k, i in enumerate(order, 1):
                d = prepared[i]
                if d["nbr"] is None:
                    try:
                        d["nbr"] = _knn_graph(d["verts_norm"], meta["k"])
                    except Exception as exc:               # noqa: BLE001
                        logger.warning("skipping part (kNN graph failed): %s", exc)
                        continue
                if k % hb == 0 or k == len(order):
                    logger.info("  epoch %d: %d/%d parts trained (%.0fs)",
                                ep, k, len(order), time.time() - ep_t0)
                n = d["x"].shape[0]
                big = device != "cpu" and max_gpu_verts and n > max_gpu_verts
                try:
                    parts = _accumulate(d, "cpu" if big else device, device, accum)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    logger.warning("CUDA OOM on %d-vertex part -> CPU fallback", n)
                    parts = _accumulate(d, "cpu", device, accum)
                if k % accum == 0 or k == len(order):
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(opt)    # unscale before clipping the real grads
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)        # unscales grads, skips step on inf/nan
                    scaler.update()         # adapt the loss scale (no-op when off)
                    opt.zero_grad(set_to_none=True)
                tot += parts["total"]
            # metric-driven schedulers (plateau) are stepped by the bookkeeper with
            # the val metric; per-epoch schedulers step here.
            if sched is not None and not sched_metric:
                sched.step()
            if bk.after_epoch(ep, tot / max(1, len(prepared))):
                logger.info("early stopping at epoch %d", ep)
                break
    except KeyboardInterrupt:
        bk.flush_history()
        logger.warning("training interrupted -- resume from the last checkpoint%s",
                       f" ({last_path})" if last_path else "")
        raise
    bk.finalize()
    return model, meta


def infer_knngraph(model, meta, verts_norm, device="cpu", max_gpu_verts=60000,
                   offset_scale=1.0, graph_cache=None):
    """Run a trained kNN-graph regressor on one part -> (N,7) numpy array ready
    for cp_targets.decode_predictions. Needs only normalised vertices (no faces).

    Density match: a part with more than the model's training cap of vertices is
    uniformly subsampled to that cap (uniform_subsample_idx -- the SAME routine
    training uses), so the kNN graph keeps the density the model trained on and the
    train/inference vertex distributions match. The cap is read from
    meta['subsample'] (what training used); the `max_gpu_verts` arg is only the
    fallback for older checkpoints that predate it. Predictions are scattered back
    to a full (N,7) array -- un-sampled vertices get heat 0 and never vote -- so
    the decode/metrics path is unchanged. CUDA OOM still falls back to CPU.

    graph_cache : optional dict to memoise the (CPU) kNN graph per vertex array,
    keyed by id(verts_norm). The validation set reuses the SAME arrays every eval,
    so this avoids rebuilding all val graphs on each metrics_fn call.

    Note: a GT point whose nearest KEPT vertex is dropped becomes unrecoverable
    (heat 0) -- inevitable when inference has no GT to protect peaks. Quantify the
    risk per corpus with subsample_coverage() / train_cp's coverage report."""
    torch = _require_torch()
    import copy
    V = np.asarray(verts_norm)
    n = len(V)
    cap = int(meta.get("subsample") or max_gpu_verts or 0)
    # fixed seed 0: inference has no part id, and only the DISTRIBUTION (uniform at
    # this density) needs to match training, not the exact vertex identities.
    idx = uniform_subsample_idx(n, cap, seed=0)
    sub = idx is not None
    if sub:
        logger.info("infer_knngraph: subsampled %d -> %d verts (%.0f%% kept)",
                    n, cap, 100.0 * cap / n)
        Vq = V[idx]
    else:
        idx = None
        Vq = V

    cache_key = id(verts_norm) if graph_cache is not None else None
    nbr_cpu = graph_cache.get(cache_key) if cache_key is not None else None
    if nbr_cpu is None:
        nbr_cpu = _knn_graph(Vq, meta["k"])
        if cache_key is not None:
            graph_cache[cache_key] = nbr_cpu

    def _run(rd):
        nbr = nbr_cpu.to(rd)
        x = torch.tensor(np.asarray(Vq), dtype=torch.float32, device=rd)
        mdl = model if rd == device else copy.deepcopy(model).to(rd)
        mdl.eval()
        with torch.no_grad():
            out = mdl(x, nbr)
        return pred_to_array(out, offset_scale=offset_scale)

    try:
        arr = _run(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        logger.warning("CUDA OOM during inference on %d-vertex part -> CPU", len(Vq))
        arr = _run("cpu")

    if not sub:
        return arr
    assert idx is not None                # bound whenever sub is True (early-returned above)
    full = np.zeros((n, N_CHANNELS), dtype=arr.dtype)
    full[idx] = arr                       # un-sampled verts stay heat=0 (no vote)
    return full


def _selftest():
    """Overfit the MLP on one synthetic plate; check it recovers the CPs."""
    torch = _require_torch()
    torch.manual_seed(0)
    np.random.seed(0)
    import json_dataset as jd
    # synthetic plate part
    xs, ys = np.meshgrid(np.linspace(0, 100, 40), np.linspace(0, 100, 40))
    V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    F = np.zeros((0, 3), dtype=np.int64)
    cp_pts = np.array([[25, 25, 0], [75, 30, 0], [50, 80, 0]], float)
    cp_dir = np.array([[0, 0, 1.0]] * 3)
    part = jd.Part("SYN", V, F, cp_pts, cp_dir, ["a", "b", "c"])
    s = prepare_sample(part, dedup=False)
    # 1200 epochs: the bce heatmap loss sharpens a lone-vertex peak slowly, so a
    # short overfit run lands the peak just under the 0.5 decode threshold (~0.48
    # at 400) and decodes nothing. An overfit smoke-test must run long enough to
    # actually overfit; centernet/focal converge faster but bce is the default.
    model = train_cpmlp([s], epochs=1200, lr=1e-3, log_every=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(s["verts_norm"], dtype=torch.float32))
    arr = pred_to_array(out, offset_scale=s["scale"])
    got = ct.decode_predictions(V, arr, heatmap_thresh=0.5, nms_radius_mm=15.0)
    print("cp_regressor selftest: recovered", len(got), "of 3 CPs")
    # each ground-truth CP should be matched by some prediction (recall=1)
    matched = 0
    for cp in cp_pts:
        d = min(np.linalg.norm(g["point"] - cp) for g in got) if got else 1e9
        if d <= 10.0:
            matched += 1
    for g in got:
        d = np.linalg.norm(cp_pts - g["point"], axis=1).min()
        assert g["direction"][2] > 0.9, "direction not outward (+z)"
        print("   point %s  locErr=%.2fmm  dir=%s" %
              (np.round(g["point"], 1), d, np.round(g["direction"], 2)))
    assert matched == 3, f"overfit MLP recovered only {matched}/3 GT CPs"
    print("cp_regressor selftest OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()

# Run a trained connection-point detector on a mesh -> robot-ready JSON.
#
# This is the inference CLI for the cp_regressor keypoint detector (the model
# train_cp.py trains). It is SEPARATE from run_on_file.py / infer_pipeline.py,
# which drive the segmentation/connector3d pipeline and cannot load this kind of
# checkpoint.
#
# Pipeline per part:
#   load model + decode params (cp_regressor.load_inference)
#       -> per-vertex (heatmap, offset, direction)  (infer_knngraph/diffusionnet/mlp)
#       -> discrete connection points               (cp_targets.decode_predictions)
#       -> robot-ready node dicts                    (cp_targets.decoded_to_robot_nodes)
#
# The decode operating point (heatmap_thresh / min_votes / nms_clearance_mm) is
# read from the artifact itself -- exactly what train_cp.py --export-model wrote,
# or DEFAULT_DECODE for an older bare checkpoint -- and can be overridden on the
# CLI for experimentation.
#
# Examples:
#   python predict.py model.pt part.json --out preds.json
#   python predict.py checkpoints/cp_knngraph_best.ckpt synthetic_data --out preds.json
#   python predict.py model.pt part.json --heatmap-thresh 0.5 --min-votes 3

import os
import json
import argparse
import logging

import numpy as np

import json_dataset as jd
import cp_targets as ct
import cp_regressor as cpr
import step_to_json as sj

logger = logging.getLogger(__name__)


def _check_units(part):
    """The whole pipeline assumes millimetres (dist_thresh_mm / nms_clearance_mm /
    approach_distance_mm are absolute mm). A part exported in metres or inches
    decodes nonsensically; warn on a wildly out-of-range bbox so the unit mistake
    is visible instead of silent (a real connector is ~1mm..1m across)."""
    V = np.asarray(part.vertices, dtype=np.float64)
    if len(V) == 0:
        return
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    if diag and (diag < 1.0 or diag > 1.0e5):
        logger.warning("part %s: bbox diagonal %.4g looks unlike millimetres -- "
                       "the decode operating point is in absolute mm, so a part in "
                       "metres/inches will decode wrong. Convert to mm first.",
                       part.part_nr, diag)


def _infer_array(backbone, model, meta, part, device, op_cache_dir, max_gpu_verts):
    """One part -> (N,7) numpy (heatmap sigmoid'd, offset in mm, direction unit)."""
    Vn, _center, scale = cpr.normalize_vertices(part.vertices)
    if backbone == "knngraph":
        return cpr.infer_knngraph(model, meta, Vn, device=device,
                                  max_gpu_verts=max_gpu_verts, offset_scale=scale)
    if backbone == "diffusionnet":
        return cpr.infer_diffusionnet(model, meta, part.vertices, part.faces, Vn,
                                      op_cache_dir=op_cache_dir, device=device,
                                      max_gpu_verts=max_gpu_verts, offset_scale=scale)
    if backbone == "mlp":
        import torch
        model.eval()
        with torch.no_grad():
            x = torch.tensor(Vn, dtype=torch.float32, device=device)
            return cpr.pred_to_array(model(x), offset_scale=scale)
    raise ValueError(f"unknown backbone {backbone!r}")


def predict_part(backbone, model, meta, part, decode, device="cpu",
                 op_cache_dir=None, max_gpu_verts=60000, approach_distance_mm=0.0,
                 snap_to_surface=False):
    """Detect connection points on one Part -> list of robot-ready node dicts."""
    _check_units(part)
    arr = _infer_array(backbone, model, meta, part, device, op_cache_dir, max_gpu_verts)
    # NMS merge radius is the fixed tool clearance, decoupled from sigma (tying it
    # to 2*sigma ballooned it on elongated parts and merged distinct CPs -- see
    # train_cp._nms_radius). Tune via --nms-clearance-mm / sweep_decode.
    nms_radius = decode["nms_clearance_mm"]
    preds = ct.decode_predictions(part.vertices, arr,
                                  heatmap_thresh=decode["heatmap_thresh"],
                                  nms_radius_mm=nms_radius,
                                  min_votes=decode["min_votes"],
                                  snap_to_surface=snap_to_surface)
    # NB: GT terminal names are unavailable at inference (using part.cp_names would
    # be label leakage), so terminal_names is intentionally left unset here.
    return ct.decoded_to_robot_nodes(preds, approach_distance_mm=approach_distance_mm), preds


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run a trained connection-point detector on a mesh -> JSON")
    ap.add_argument("model", nargs="?",
                    help="exported .pt or any cp_regressor checkpoint "
                         "(omit when using --from-cad)")
    ap.add_argument("source", help="a part .json, a top-level-array .json, a raw 3D "
                                    "model (.stp/.step/.stl/.obj/.off), or a directory of any of these")
    ap.add_argument("--deflection", type=float, default=0.5,
                    help="gmsh max mesh size (mm) when tessellating a STEP input "
                         "(only used for .stp/.step sources)")
    ap.add_argument("--from-cad", action="store_true",
                    help="for STEP input, extract connection points DIRECTLY from the "
                         "CAD cylindrical openings (step_openings) instead of the ML "
                         "model -- exact, no model needed (the right choice when the "
                         ".stp is available)")
    ap.add_argument("--cad-rmin", type=float, default=1.3,
                    help="--from-cad: min opening radius (mm) to keep")
    ap.add_argument("--cad-rmax", type=float, default=3.0,
                    help="--from-cad: max opening radius (mm) to keep")
    ap.add_argument("--cad-auto", action="store_true",
                    help="--from-cad: auto-pick the terminal-hole radius per part "
                         "(no manual --cad-rmin/--cad-rmax)")
    ap.add_argument("--out", default=None,
                    help="write predictions JSON here (default: stdout summary only)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--op-cache-dir", default=None,
                    help="DiffusionNet operator cache (diffusionnet backbone only)")
    ap.add_argument("--max-gpu-verts", type=int, default=60000)
    ap.add_argument("--approach-distance-mm", type=float, default=0.0,
                    help="standoff distance along the outward approach vector")
    ap.add_argument("--snap-to-surface", action="store_true",
                    help="snap each detected point onto the nearest mesh vertex "
                         "(decoded points are vote means and may float off-surface)")
    # decode overrides -- default None means 'use the value bundled in the model'
    ap.add_argument("--heatmap-thresh", type=float, default=None)
    ap.add_argument("--min-votes", type=int, default=None)
    ap.add_argument("--nms-clearance-mm", type=float, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # CAD-direct path: extract openings straight from the STEP (no ML model).
    if args.from_cad:
        import step_openings as so
        res = so.run(args.source, rmin=args.cad_rmin, rmax=args.cad_rmax,
                     auto=args.cad_auto)
        for p in res["parts"]:
            logger.info("  %-28s %d connection point(s)  (from CAD)",
                        p["part_nr"], p["n_detected"])
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2)
            logger.info("predictions written to %s", args.out)
        else:
            print(json.dumps(res, indent=2))
        return

    if args.model is None:
        ap.error("model is required unless --from-cad is given")
    model, meta, backbone, decode = cpr.load_inference(args.model, device=args.device)
    # CLI overrides win over the bundled operating point
    if args.heatmap_thresh is not None:
        decode["heatmap_thresh"] = args.heatmap_thresh
    if args.min_votes is not None:
        decode["min_votes"] = args.min_votes
    if args.nms_clearance_mm is not None:
        decode["nms_clearance_mm"] = args.nms_clearance_mm
    logger.info("decode operating point: %s", decode)

    results = []
    n_parts = n_points = n_skipped = 0
    for part in sj.iter_parts_any(args.source, deflection=args.deflection):
        # one malformed/degenerate part must not abort the whole batch
        try:
            nodes, preds = predict_part(
                backbone, model, meta, part, decode, device=args.device,
                op_cache_dir=args.op_cache_dir, max_gpu_verts=args.max_gpu_verts,
                approach_distance_mm=args.approach_distance_mm,
                snap_to_surface=args.snap_to_surface)
        except Exception as exc:                              # noqa: BLE001
            n_skipped += 1
            logger.warning("  %-20s SKIPPED (inference failed): %s",
                           getattr(part, "part_nr", "?"), exc)
            results.append({"part_nr": getattr(part, "part_nr", "?"),
                            "n_vertices": int(getattr(part, "n_vertices", 0)),
                            "error": str(exc), "connection_points": []})
            continue
        n_parts += 1
        n_points += len(nodes)
        logger.info("  %-20s verts=%-6d  detected %d connection point(s)",
                    part.part_nr, part.n_vertices, len(nodes))
        results.append({"part_nr": part.part_nr,
                        "n_vertices": int(part.n_vertices),
                        "n_detected": len(nodes),
                        "connection_points": nodes})

    if n_parts == 0 and n_skipped == 0:
        raise SystemExit(f"no parts found in {args.source}")
    logger.info("done: %d part(s), %d connection point(s) total%s", n_parts, n_points,
                ("  (%d skipped)" % n_skipped) if n_skipped else "")

    payload = {"backbone": backbone, "decode": decode,
               "n_parts": n_parts, "n_detected_total": n_points,
               "n_skipped": n_skipped, "parts": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("predictions written to %s", args.out)
    else:
        # no --out: still emit machine-readable JSON to stdout for piping
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

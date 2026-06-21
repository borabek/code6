# Extract connection openings DIRECTLY from a STEP B-rep -- NO mesh tessellation
# loss, NO ML -- using gmsh's CAD kernel so the coordinates match the tessellated
# mesh exactly (raw STEP placements are in local frames; gmsh applies the
# assembly transform, giving cylinders in the same frame as step_to_json's mesh).
#
# On these connectors the circular openings are real CAD features
# (CYLINDRICAL_SURFACE). gmsh reports each surface's type, centre of mass and
# bounding box; cylinders of terminal radius -> connection points (centre +
# radius). A hole is several cylindrical faces, so coincident ones are merged.
#
# This is the cleanest "real label" source when the STEP is available.
#
# Usage:
#   python step_openings.py part.stp --out openings.json
#   python step_openings.py part.stp --out o.json --rmin 1.3 --rmax 3.0
#   python step_openings.py "3d models" --out o.json
#   python step_openings.py part.stp --list            # radius histogram only

import os
import json
import glob
import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cyl_radius(ext):
    """Radius of an axis-aligned cylinder from its bbox extents: the two CLOSEST
    extents are the two diameters (2r); the third is the height."""
    s = np.sort(np.asarray(ext, float))
    pair = (s[0], s[1]) if (s[1] - s[0]) <= (s[2] - s[1]) else (s[1], s[2])
    return float((pair[0] + pair[1]) / 4.0)


def _cyl_axis_dim(ext):
    """Bbox dimension along the cylinder axis = the 'odd-one-out' extent (height);
    the two close extents are the diameter."""
    o = np.argsort(np.asarray(ext, float))
    s = np.asarray(ext, float)[o]
    return int(o[2]) if (s[1] - s[0]) <= (s[2] - s[1]) else int(o[0])


def auto_radius_range(radii, fillet_floor=1.3, split_gap=1.0):
    """Auto-pick (rmin, rmax) for the terminal holes, so a new connector family
    needs no manual --rmin/--rmax. Edge fillets/rounds cluster below ~1 mm and are
    dropped (fillet_floor). A connector often has SEVERAL terminal hole sizes plus
    a few much larger mounting/structural holes; the terminal cluster is separated
    from those by a big jump in radius, so we keep everything from the floor up to
    the radius just below the largest gap (> split_gap mm). With no clear gap, keep
    all holes above the floor. Falls back to 1.0-5.0 mm if nothing is above it."""
    r = sorted(set(round(float(x), 1) for x in radii if x >= fillet_floor))
    if not r:
        return 1.0, 5.0
    if len(r) == 1:
        return max(0.0, r[0] - 0.3), r[0] + 0.3
    gaps = [(r[i + 1] - r[i], i) for i in range(len(r) - 1)]
    biggest, k = max(gaps)
    rmax = (r[k] if biggest > split_gap else r[-1]) + 0.3
    return fillet_floor - 0.1, rmax


def extract_cylinders(path):
    """All cylindrical faces of a STEP -> (cyls, part_bbox) in the gmsh
    (== tessellated-mesh) coordinate frame. cyls = [{center, radius, axis_dim, ext}];
    part_bbox = (xmin,ymin,zmin,xmax,ymax,zmax)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        cyls = []
        for (d, t) in gmsh.model.getEntities(2):
            if gmsh.model.getType(d, t) != "Cylinder":
                continue
            com = np.array(gmsh.model.occ.getCenterOfMass(d, t), dtype=float)
            bb = gmsh.model.getBoundingBox(d, t)
            ext = np.array([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
            cyls.append({"center": com, "radius": _cyl_radius(ext),
                         "axis_dim": _cyl_axis_dim(ext), "ext": ext})
        bbox = np.array(gmsh.model.getBoundingBox(-1, -1), dtype=float)
        return cyls, bbox
    finally:
        gmsh.finalize()


def openings_from_step(path, rmin=1.3, rmax=3.0, merge_tol=4.0, auto=False,
                       drop_corner=0.0):
    """Cylinders of terminal radius -> merged openings (centre + axis + radius).

    auto=True picks the radius band per part (auto_radius_range). drop_corner>0
    (EXPERIMENTAL, opt-in): drop holes whose in-face position is within this
    fraction of the part edge -- likely mounting/corner holes. RISKY: connector
    terminals are often near an edge too, so it can drop real CPs; default off."""
    cyls, bbox = extract_cylinders(path)
    if auto:
        rmin, rmax = auto_radius_range([c["radius"] for c in cyls])
    keep = [c for c in cyls if rmin <= c["radius"] <= rmax]
    holes = []
    for c in keep:
        for h in holes:
            if np.linalg.norm(c["center"] - h["center"]) < merge_tol:
                h["members"].append(c)
                break
        else:
            holes.append({"center": c["center"].copy(), "members": [c]})
    if not holes:
        return []
    part_center = np.mean([h["center"] for h in holes], axis=0)
    ops = []
    for h in holes:
        c = np.mean([m["center"] for m in h["members"]], axis=0)
        r = float(np.mean([m["radius"] for m in h["members"]]))
        ad = int(np.bincount([m["axis_dim"] for m in h["members"]], minlength=3).argmax())
        face_dims = [i for i in range(3) if i != ad]
        # #2 corner / mounting-hole filter (opt-in): skip holes hugging a face edge
        if drop_corner > 0.0:
            edge = False
            for fd in face_dims:
                span = bbox[fd + 3] - bbox[fd]
                if span > 0 and min(c[fd] - bbox[fd], bbox[fd + 3] - c[fd]) / span < drop_corner:
                    edge = True
            if edge:
                continue
        # #3 insert-direction sign: a hole sitting clearly nearer ONE face along its
        # axis (a blind hole / recess) is entered from that face; a symmetric
        # through-hole falls back to pointing away from the part centroid.
        lo, hi = bbox[ad], bbox[ad + 3]
        span = hi - lo
        d_lo, d_hi = c[ad] - lo, hi - c[ad]
        if span > 0 and abs(d_hi - d_lo) > 0.15 * span:
            sign = 1.0 if d_hi < d_lo else -1.0          # toward the nearer (open) face
        else:
            sign = 1.0 if (c[ad] - part_center[ad]) >= 0 else -1.0
        direction = np.zeros(3)
        direction[ad] = sign
        ops.append({"entry_point": [float(x) for x in c],
                    "approach_vector": [float(x) for x in direction],
                    "radius_mm": r, "n_faces": len(h["members"])})
    return ops


def run(source, rmin=1.3, rmax=3.0, merge_tol=4.0, auto=False, drop_corner=0.0):
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "**", "*.st*p"), recursive=True))
    else:
        files = [source]
    parts = []
    for f in files:
        if os.path.splitext(f)[1].lower() not in (".stp", ".step"):
            continue
        part_nr = os.path.splitext(os.path.basename(f))[0]
        try:                                              # one bad STEP must not abort the batch
            ops = openings_from_step(f, rmin=rmin, rmax=rmax, merge_tol=merge_tol,
                                     auto=auto, drop_corner=drop_corner)
        except Exception as exc:                          # noqa: BLE001
            logger.warning("  %-30s SKIPPED (unreadable STEP): %s", part_nr, exc)
            continue
        logger.info("  %-30s %d opening(s)", part_nr, len(ops))
        parts.append({"part_nr": part_nr, "n_detected": len(ops),
                      "connection_points": ops})
    return {"detector": "step_cylinders_gmsh",
            "params": {"rmin": rmin, "rmax": rmax, "merge_tol": merge_tol,
                       "auto": auto, "drop_corner": drop_corner},
            "parts": parts}


def to_abb_labels(openings):
    """Detected openings -> ABB ConnectionPoints (for fine-tuning the ML model)."""
    return [{"Index": i, "Name": f"OPEN-{i}",
             "Point": {"X": o["entry_point"][0], "Y": o["entry_point"][1],
                       "Z": o["entry_point"][2]},
             "InsertDirection": {"X": o["approach_vector"][0],
                                 "Y": o["approach_vector"][1],
                                 "Z": o["approach_vector"][2]}}
            for i, o in enumerate(openings)]


def _geom_sig(V, n_ops):
    """Coarse geometry fingerprint (bbox dims in mm + vertex + CP count) to catch
    near-identical re-downloads of the same part."""
    if not len(V):
        return ("empty", n_ops)
    return (tuple(np.round(V.max(0) - V.min(0), 0).tolist()), len(V), n_ops)


def write_labeled_corpus(source, out_dir, rmin=1.3, rmax=3.0, merge_tol=4.0,
                         deflection=0.5, auto=False, drop_corner=0.0, dedup=True):
    """Turn STEP file(s) into a REAL LABELLED ABB-JSON corpus: the gmsh mesh plus
    the cylinder-extracted connection points as ground-truth ConnectionPoints.
    The mesh (step_to_json, gmsh tessellation) and the CPs (step_openings, gmsh
    OCC) share the gmsh coordinate frame, so the labels sit exactly on the mesh.

    This is the #2 bridge: it converts unlabelled .stp parts into data you can
    train / fine-tune `train_cp.py` on. dedup=True skips a part whose geometry
    matches one already written (near-identical re-downloads would otherwise leak
    across the train/val split and waste training). Returns the written paths."""
    import step_to_json as sj
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "**", "*.st*p"), recursive=True))
    else:
        files = [source]
    written, seen = [], {}
    for f in files:
        if os.path.splitext(f)[1].lower() not in (".stp", ".step"):
            continue
        part_nr = os.path.splitext(os.path.basename(f))[0]
        try:
            V, F = sj.load_any_mesh(f, deflection=deflection)
            ops = openings_from_step(f, rmin=rmin, rmax=rmax, merge_tol=merge_tol,
                                     auto=auto, drop_corner=drop_corner)
        except Exception as exc:                              # noqa: BLE001
            logger.warning("  SKIPPED %s: %s", part_nr, exc)
            continue
        if dedup:
            sig = _geom_sig(V, len(ops))
            if sig in seen:
                logger.warning("  %-30s DUPLICATE of %s -- skipped", part_nr, seen[sig])
                continue
            seen[sig] = part_nr
        d = sj.mesh_to_abb_dict(part_nr, V, F)
        d["ConnectionPoints"] = to_abb_labels(ops)
        out = os.path.join(out_dir, part_nr + ".json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        logger.info("  %-30s %d verts, %d labelled CP(s) -> %s",
                    part_nr, len(V), len(ops), os.path.basename(out))
        written.append(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract connection openings from a STEP B-rep via gmsh "
                    "(cylindrical faces) -> CP labels in the mesh frame")
    ap.add_argument("source", help="a .stp/.step file or a directory of them")
    ap.add_argument("--out", help="output predictions JSON")
    ap.add_argument("--rmin", type=float, default=1.3, help="min hole radius (mm)")
    ap.add_argument("--rmax", type=float, default=3.0, help="max hole radius (mm)")
    ap.add_argument("--merge-tol", type=float, default=4.0,
                    help="merge cylindrical faces whose centres are within this (mm)")
    ap.add_argument("--list", action="store_true",
                    help="print the radius histogram of all cylinders and exit")
    ap.add_argument("--auto", action="store_true",
                    help="auto-pick the terminal-hole radius band per part (drops "
                         "edge fillets, keeps the most frequent hole radius) -- no "
                         "manual --rmin/--rmax needed for a new connector family")
    ap.add_argument("--drop-corner", type=float, default=0.0,
                    help="EXPERIMENTAL (opt-in): drop holes within this fraction of "
                         "the part edge (likely mounting holes). RISKY -- can also "
                         "drop real edge terminals; default 0 (off)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="--label-corpus: keep near-identical re-downloaded parts "
                         "(default skips geometric duplicates)")
    ap.add_argument("--label-corpus", metavar="OUTDIR",
                    help="write a REAL labelled ABB-JSON corpus (mesh + extracted CPs) "
                         "here, ready to train/fine-tune train_cp.py on")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    src = args.source
    if args.label_corpus:
        written = write_labeled_corpus(src, args.label_corpus, rmin=args.rmin,
                                       rmax=args.rmax, merge_tol=args.merge_tol,
                                       auto=args.auto, drop_corner=args.drop_corner,
                                       dedup=not args.keep_duplicates)
        print(f"\nwrote {len(written)} labelled part(s) -> {args.label_corpus}")
        print("fine-tune on them:  python train_cp.py %s --backbone knngraph "
              "--device cuda --split-group none --epochs 200 --run-name cp_real"
              % args.label_corpus)
        return
    if args.list:
        f = src
        if os.path.isdir(f):
            f = sorted(glob.glob(os.path.join(f, "**", "*.st*p"), recursive=True))[0]
        cyls, _ = extract_cylinders(f)
        rad = np.round([c["radius"] for c in cyls], 1)
        u, ct = np.unique(rad, return_counts=True)
        print("cylinder radii (count x radius_mm):")
        for r, c in zip(u, ct):
            print(f"  {c:3d} x {r:.1f}")
        return

    res = run(src, rmin=args.rmin, rmax=args.rmax, merge_tol=args.merge_tol,
              auto=args.auto, drop_corner=args.drop_corner)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    tot = sum(p["n_detected"] for p in res["parts"])
    print(f"\n{len(res['parts'])} part(s), {tot} opening(s) -> {args.out or '(stdout)'}")


def _selftest():
    """Build a STEP box with two drilled holes (gmsh) -> the detector finds two."""
    import tempfile
    import shutil
    import gmsh
    work = tempfile.mkdtemp(prefix="stepcp_")
    path = os.path.join(work, "selftest_box.step")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        box = gmsh.model.occ.addBox(0, 0, 0, 40, 20, 8)
        c1 = gmsh.model.occ.addCylinder(12, 10, -1, 0, 0, 10, 2.0)
        c2 = gmsh.model.occ.addCylinder(28, 10, -1, 0, 0, 10, 2.0)
        gmsh.model.occ.cut([(3, box)], [(3, c1), (3, c2)])
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    try:
        ops = openings_from_step(path, rmin=1.0, rmax=3.0, merge_tol=4.0)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    assert len(ops) == 2, f"expected 2 drilled holes, got {len(ops)}"
    print("step_openings selftest OK: found", len(ops), "hole(s) of r~2mm")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()

#!/usr/bin/env python3
"""
Work out what a set of trajectory coordinates actually means.

    python diagnose.py CT.nii.gz needles.json [target_mask.nii.gz]

Three things a triple of numbers might be, each crossed with all 48 axis
permutations and sign flips:

  world     NIfTI world mm, inverted through the full affine (origin included)
  index-mm  mm from the array corner: voxel = value / spacing. This is what
            code of the form mask(points, spacing, shape) implies, because
            spacing alone carries no origin and no axis direction.
  voxel     already a voxel index

Candidates are ranked by whether every target lands inside the volume, inside
the target mask if you pass one, and in soft tissue. On a pre-procedure CT
there is no metal to key on, so this narrows the field rather than deciding
it — usually to a handful. Pick between those in the viewer's frame picker,
where you can see whether the trajectories hit the tumour and exit through
the ribs.
"""
import itertools
import json
import sys

import numpy as np
import nibabel as nib

AXES = "xyz"
METAL_HU = 800


def load_points(path):
    data = json.load(open(path))
    items = data["needles"] if isinstance(data, dict) else data
    entries, targets, labels = [], [], []
    for i, n in enumerate(items):
        if "entry" in n and "target" in n:
            a, b = n["entry"], n["target"]
        else:
            pts = n["world"] if "world" in n else n["voxel"]
            a, b = pts[0], pts[-1]
        entries.append(a)
        targets.append(b)
        labels.append(n.get("label", f"Needle {i+1}"))
    return np.array(entries, float), np.array(targets, float), labels


def sign_perms():
    out = [("as given", np.eye(3))]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for row, (p, s) in enumerate(zip(perm, signs)):
                M[row, p] = s
            if np.allclose(M, np.eye(3)):
                continue
            name = " ".join(f"{'-' if s < 0 else '+'}{AXES[p]}"
                            for p, s in zip(perm, signs))
            out.append((name, M))
    return out


def models(affine, spacing):
    """How a triple of numbers might map to a file-order voxel index.

    Two index-mm variants exist because the axis order of the array and the
    axis order of the spacing vector need not agree. If the loader transposes
    the volume (out = out.transpose()) but spacing is still read from the
    header in file order, the divisor is permuted relative to the index.
    """
    inv = np.linalg.inv(affine)
    sp = np.asarray(spacing, float)
    return [
        ("world",        lambda p, M: (np.c_[p @ M.T, np.ones(len(p))] @ inv.T)[:, :3]),
        ("index-mm",     lambda p, M: (p @ M.T) / sp),
        ("index-mm/src", lambda p, M: (p @ M.T) / (np.abs(M) @ sp)),
        ("voxel",        lambda p, M: p @ M.T),
    ]


def sample_line(a, b, to_vox, shape, data, n=240):
    ts = np.linspace(0, 1, n)
    pts = a + (b - a) * ts[:, None]
    vox = to_vox(pts)
    inside = np.all((vox >= 0) & (vox <= np.array(shape) - 1), axis=1)
    if not inside.any():
        return 0.0, None
    idx = np.round(vox[inside]).astype(int)
    return inside.mean(), data[idx[:, 0], idx[:, 1], idx[:, 2]]


def score(entries, targets, to_vox, shape, data):
    tvox = to_vox(targets)
    t_in = np.all((tvox >= 0) & (tvox <= np.array(shape) - 1), axis=1)
    frac_targets = t_in.mean()

    med_t = np.nan
    if t_in.any():
        idx = np.round(tvox[t_in]).astype(int)
        med_t = float(np.median(data[idx[:, 0], idx[:, 1], idx[:, 2]]))

    fracs = []
    for a, b in zip(entries, targets):
        _, vals = sample_line(a, b, to_vox, shape, data)
        if vals is not None and len(vals):
            fracs.append(float((vals > METAL_HU).mean()))
    return frac_targets, med_t, (float(np.median(fracs)) if fracs else 0.0)


def main(ct_path, needles_path, mask_path=None):
    img = nib.load(ct_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    shape = img.shape[:3]
    spacing = img.header.get_zooms()[:3]
    mask = None
    if mask_path:
        mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
        if mask.shape[:3] != tuple(shape):
            print(f"warning: mask shape {mask.shape[:3]} != CT shape {tuple(shape)}; ignoring")
            mask = None

    corners = np.array([[i, j, k, 1] for i in (0, shape[0] - 1)
                        for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])
    w = (corners @ img.affine.T)[:, :3]
    print(f"CT   {ct_path}")
    print(f"     {shape[0]}x{shape[1]}x{shape[2]} voxels, spacing "
          f"{spacing[0]:.3g} x {spacing[1]:.3g} x {spacing[2]:.3g} mm")
    print(f"     world  x {w[:,0].min():7.1f}..{w[:,0].max():7.1f}"
          f"   y {w[:,1].min():7.1f}..{w[:,1].max():7.1f}"
          f"   z {w[:,2].min():7.1f}..{w[:,2].max():7.1f}")

    entries, targets, _ = load_points(needles_path)
    rows = []
    for mname, fn in models(img.affine, spacing):
        for pname, M in sign_perms():
            v = fn(targets, M)
            if not np.all((v >= 0) & (v <= np.array(shape) - 1)):
                continue
            idx = np.round(v).astype(int)
            hu = data[idx[:, 0], idx[:, 1], idx[:, 2]]
            tissue = float(np.mean((hu > -150) & (hu < 300)))
            inmask = float(mask[idx[:, 0], idx[:, 1], idx[:, 2]].mean()) if mask is not None else None
            rows.append((inmask, tissue, mname, pname))

    if not rows:
        print("\nNo candidate puts every target inside the volume.")
        print("The coordinates probably belong to a different scan.")
        return

    rows.sort(key=lambda r: (-(r[0] or 0), -r[1]))
    hdr = "in mask" if mask is not None else ""
    print(f"\n{'model':<13} {'axes':<12} {hdr:>8} {'in tissue':>10}")
    print("-" * 48)
    for inmask, tissue, mname, pname in rows[:10]:
        mm = "—" if inmask is None else f"{inmask*100:.0f}%"
        print(f"{mname:<13} {pname:<12} {mm:>8} {tissue*100:9.0f}%")

    keep = [r for r in rows if (r[0] == 1.0 if mask is not None else r[1] == 1.0)]
    print()
    if not keep:
        print("Nothing places every target in the "
              + ("mask" if mask is not None else "soft tissue") + ".")
    elif len(keep) == 1:
        print(f'Only one candidate fits: needleFrame "{keep[0][2]}", '
              f'needleTransform "{keep[0][3]}"')
    else:
        print(f"{len(keep)} candidates fit equally well:")
        for r in keep[:6]:
            print(f'   needleFrame "{r[2]}"  needleTransform "{r[3]}"')
        print("\nThey cannot be separated from the image alone. Open the viewer,")
        print("use the Coordinate frame picker, and choose the one whose needles")
        print("run from the ribs to the tumour.")
    if mask is None:
        print("\nPass a target mask as a third argument to narrow this considerably.")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        raise SystemExit(1)
    main(*sys.argv[1:])

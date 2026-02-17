#!/usr/bin/env python3
"""
simulate_observer_masks.py

Simulate observer-specific segmentation masks from ground-truth segmentations, this is done in three steps:
  1) Boundary perturbation (BSpline warp + directional field aligned to SDF gradient
     with observer-specific COM shifts),
  2) Edge-aware SDM noise (Sobel-derived image contrast uncertainty),
  3) Volume perturbation  (SDF thresholding using observer-specific volume ratio).

Expected dataset structure (BraTS MEN-style):
  <ground_truth_dir>/
    BraTS-MEN-XXXXX-000/
      BraTS-MEN-XXXXX-000-t1c.nii.gz
      BraTS-MEN-XXXXX-000-seg.nii.gz
    ...

Outputs:
  <output_dir>/
    BraTS-MEN-XXXXX-000_observer_Observer1.nii.gz
or if --numeric-observer-id:
    BraTS-MEN-XXXXX-000_observer_1.nii.gz

Dependencies:
  numpy, scipy, SimpleITK
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter


# ---------------------------- Observer-specific parameters (taken from literature) ---------------------------- #

VOLUME_RATIO_RANGES: Dict[str, Tuple[float, float]] = {
    "Observer1": (0.88, 1.46),
    "Observer2": (0.85, 1.17),
    "Observer3": (0.92, 1.12),
    "Observer4": (0.59, 1.16),
    "Observer5": (1.07, 1.55),
    "Observer6": (0.85, 1.09),
    "Observer7": (0.64, 1.12),
    "Observer8": (0.92, 1.03),
    "Observer9": (0.95, 1.08),
}

COM_VALUES = np.array(
    [
        [2.0, 4.9, 4.0],
        [0.6, 2.2, 2.3],
        [11.0, 7.9, 3.3],
        [5.9, 8.0, 6.5],
        [4.2, 3.2, 1.5],
        [6.2, 1.0, 2.0],
        [1.2, 3.2, 5.0],
        [2.2, 1.8, 4.4],
        [3.0, 2.5, 6.5],
    ],
    dtype=np.float32,
)


# ------------------------------- Data model -------------------------------- #

@dataclass(frozen=True)
class CasePaths:
    case_id: str
    t1c_path: str
    seg_path: str


# ------------------------------- Logging ----------------------------------- #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


# ------------------------------- Functions ---------------------------------- #

def compute_signed_distance(mask_zyx: np.ndarray) -> np.ndarray:
    """Signed distance: positive inside mask, negative outside."""
    m = mask_zyx.astype(bool)
    return distance_transform_edt(m) - distance_transform_edt(~m)


def normalize_volume(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def threshold_sdf_masks(
    sdf: np.ndarray,
    min_ratio: float,
    max_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Threshold SDF to match target volume ratio sampled from [min_ratio, max_ratio]."""
    original_volume = int(np.sum(sdf >= 0))
    if original_volume <= 0:
        return np.zeros_like(sdf, dtype=np.uint8), float("nan")

    target_ratio = float(rng.uniform(min_ratio, max_ratio))
    target_volume = int(original_volume * target_ratio)
    target_volume = max(1, min(target_volume, sdf.size))

    flat = sdf.ravel()
    order = np.argsort(flat)[::-1]  # descending
    threshold_value = float(flat[order[target_volume - 1]])
    new_mask = (sdf >= threshold_value).astype(np.uint8)
    return new_mask, threshold_value


def compute_edge_uncertainty_map(t1c_path: str) -> np.ndarray:
    """
    Sobel edges -> normalize -> uncertainty = 1 - normalized edge magnitude.
    Higher values indicate greater uncertainty in the segmentation task.
    """
    img = sitk.ReadImage(t1c_path)
    edge = sitk.SobelEdgeDetection(img)
    edge_np = sitk.GetArrayFromImage(edge)
    norm_edge = normalize_volume(edge_np)
    return 1.0 - norm_edge


def apply_sdm_perturbation_with_uncertainty(
    sdf: np.ndarray,
    uncertainty_map: np.ndarray,
    rng: np.random.Generator,
    base_std: float = 0.5,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Add spatially-smoothed noise modulated by uncertainty map."""
    noise = rng.normal(0.0, 1.0, size=sdf.shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=smooth_sigma)
    modulated = noise * (uncertainty_map.astype(np.float32) * float(base_std))
    return sdf + modulated


def directional_bias(seg_path: str, com_mean_shift_xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Directional bias field aligned with unit SDF gradient.
    bias_magnitude is sampled per-axis around observer COM mean shift.
    """
    seg_img = sitk.ReadImage(seg_path)
    seg_np = (sitk.GetArrayFromImage(seg_img) > 0).astype(np.uint8)

    sdf = compute_signed_distance(seg_np)

    grad_z, grad_y, grad_x = np.gradient(sdf.astype(np.float32))
    gradient = np.stack([grad_x, grad_y, grad_z], axis=-1)  # xyz vector at each voxel

    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    unit_gradient = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm != 0)

    shape = unit_gradient.shape[:-1]
    # per-axis magnitudes
    bias_mag = np.stack(
        [
            rng.normal(loc=float(com_mean_shift_xyz[0]), scale=1.0, size=shape),
            rng.normal(loc=float(com_mean_shift_xyz[1]), scale=1.0, size=shape),
            rng.normal(loc=float(com_mean_shift_xyz[2]), scale=1.0, size=shape),
        ],
        axis=-1,
    ).astype(np.float32)

    return unit_gradient * bias_mag


def apply_directional_bias_to_BSpline(
    transform: sitk.BSplineTransform,
    directional_bias_field: np.ndarray,
    reference_img: sitk.Image,
) -> sitk.BSplineTransform:
    """
    Populate BSpline control-point parameters by sampling the directional bias field.
    directional_bias_field is indexed as [z, y, x, 3] with xyz displacements.
    """
    mesh_size = transform.GetTransformDomainMeshSize()
    size_xyz = reference_img.GetSize()  # (x, y, z)
    control_points = [m + 3 for m in mesh_size]

    displacement_params: List[float] = []

    # NOTE: we sample in image index space
    for z in np.linspace(0, size_xyz[2] - 1, control_points[2]):
        for y in np.linspace(0, size_xyz[1] - 1, control_points[1]):
            for x in np.linspace(0, size_xyz[0] - 1, control_points[0]):
                zi, yi, xi = int(round(z)), int(round(y)), int(round(x))

                # directional_bias_field shape is [z, y, x, 3]
                zi = max(0, min(zi, directional_bias_field.shape[0] - 1))
                yi = max(0, min(yi, directional_bias_field.shape[1] - 1))
                xi = max(0, min(xi, directional_bias_field.shape[2] - 1))

                disp_xyz = directional_bias_field[zi, yi, xi]
                displacement_params.extend([float(disp_xyz[0]), float(disp_xyz[1]), float(disp_xyz[2])])

    transform.SetParameters(displacement_params)
    return transform


def boundary_perturbations(
    t1c_path: str,
    seg_path: str,
    com_mean_shift_xyz: np.ndarray,
    rng: np.random.Generator,
    mesh_size: int = 8,
) -> np.ndarray:
    """Apply BSpline warp to segmentation based on directional bias field."""
    seg_img = sitk.ReadImage(seg_path)
    transform = sitk.BSplineTransformInitializer(seg_img, [mesh_size] * seg_img.GetDimension(), order=3)

    bias_field = directional_bias(seg_path, com_mean_shift_xyz, rng=rng)
    transform = apply_directional_bias_to_BSpline(transform, bias_field, seg_img)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(seg_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)

    warped_seg = resampler.Execute(seg_img)
    warped_np = (sitk.GetArrayFromImage(warped_seg) > 0).astype(np.uint8)
    return warped_np


def discover_cases(root_dir: str) -> List[CasePaths]:
    """Find cases under root_dir that contain *-t1c.nii.gz and *-seg.nii.gz."""
    cases: List[CasePaths] = []

    for folder in sorted(os.listdir(root_dir)):
        case_dir = os.path.join(root_dir, folder)
        if not os.path.isdir(case_dir):
            continue

        t1c_path: Optional[str] = None
        seg_path: Optional[str] = None

        for f in os.listdir(case_dir):
            if f.endswith("-t1c.nii.gz"):
                t1c_path = os.path.join(case_dir, f)
            elif f.endswith("-seg.nii.gz"):
                seg_path = os.path.join(case_dir, f)

        if t1c_path and seg_path:
            case_id = os.path.basename(t1c_path).replace("-t1c.nii.gz", "")
            cases.append(CasePaths(case_id=case_id, t1c_path=t1c_path, seg_path=seg_path))

    return cases


def save_mask(mask_zyx: np.ndarray, reference_seg_path: str, out_path: str) -> None:
    """Save a (z,y,x) uint8 mask with spatial metadata copied from reference seg."""
    ref = sitk.ReadImage(reference_seg_path)
    out = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, out_path)


def observer_output_id(observer_id: str, numeric: bool) -> str:
    if not numeric:
        return observer_id
    digits = "".join(filter(str.isdigit, observer_id))
    return digits if digits else observer_id


# ------------------------------- CLI ---------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulate observer masks (BraTS MEN).")
    p.add_argument("--ground-truth-dir", required=True, help="Root dir containing per-case folders.")
    p.add_argument("--output-dir", required=True, help="Output directory for simulated masks.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--mesh-size", type=int, default=8, help="BSpline mesh size per dimension.")
    p.add_argument("--base-std", type=float, default=0.5, help="Base noise std for SDM perturbation.")
    p.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma for smoothing noise.")
    p.add_argument(
        "--numeric-observer-id",
        action="store_true",
        help="Use numeric observer IDs in filenames (Observer1 -> 1).",
    )
    p.add_argument(
        "--observers",
        nargs="*",
        default=None,
        help="Subset of observers to simulate (e.g., Observer1 Observer4). Default: all.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.verbose)

    os.makedirs(args.output_dir, exist_ok=True)

    cases = discover_cases(args.ground_truth_dir)
    if not cases:
        raise FileNotFoundError(
            f"No cases found under {args.ground_truth_dir}. "
            "Expected folders containing *-t1c.nii.gz and *-seg.nii.gz."
        )

    all_observers = list(VOLUME_RATIO_RANGES.keys())
    if args.observers:
        requested = set(args.observers)
        missing = requested.difference(all_observers)
        if missing:
            raise ValueError(f"Unknown observers: {sorted(missing)}")
        observers = [o for o in all_observers if o in requested]
    else:
        observers = all_observers

    rng = np.random.default_rng(args.seed)

    logging.info("Found %d cases.", len(cases))
    logging.info("Output dir: %s", args.output_dir)
    logging.info("Observers: %s", ", ".join(observers))

    # Observer index mapping: Observer1 -> row0, Observer2 -> row1, ...
    def com_for_observer(obs: str) -> np.ndarray:
        idx = int(obs.replace("Observer", "")) - 1
        return COM_VALUES[idx]

    n_written = 0

    for obs in observers:
        min_r, max_r = VOLUME_RATIO_RANGES[obs]
        com_shift = com_for_observer(obs)

        logging.info("Simulating %s (ratio %.2f-%.2f, COM %s)", obs, min_r, max_r, com_shift.tolist())

        for case in cases:
            # Step 1: Boundary bias via BSpline + COM directional field
            warped_mask = boundary_perturbations(
                t1c_path=case.t1c_path,
                seg_path=case.seg_path,
                com_mean_shift_xyz=com_shift,
                rng=rng,
                mesh_size=args.mesh_size,
            )

            # Step 2: Convert to SDF
            sdf = compute_signed_distance(warped_mask)

            # Step 3: Edge-aware SDM noise
            uncertainty = compute_edge_uncertainty_map(case.t1c_path)
            sdf = apply_sdm_perturbation_with_uncertainty(
                sdf=sdf,
                uncertainty_map=uncertainty,
                rng=rng,
                base_std=args.base_std,
                smooth_sigma=args.smooth_sigma,
            )

            # Step 4: Volume perturbation
            adjusted_mask, thr = threshold_sdf_masks(sdf, min_r, max_r, rng=rng)

            if adjusted_mask.sum() == 0:
                logging.warning("Empty simulated mask for %s %s; using warped mask fallback.", case.case_id, obs)
                adjusted_mask = warped_mask

            # Step 5: Save output
            obs_out = observer_output_id(obs, numeric=args.numeric_observer_id)
            out_name = f"{case.case_id}_observer_{obs_out}.nii.gz"
            out_path = os.path.join(args.output_dir, out_name)

            save_mask(adjusted_mask, case.seg_path, out_path)
            n_written += 1

            if args.verbose >= 2:
                logging.debug("Wrote %s (thr=%.4f, voxels=%d)", out_name, thr, int(adjusted_mask.sum()))

    logging.warning("Done. Wrote %d masks to %s", n_written, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
simulate_observer_masks.py

Simulate observer-specific segmentation masks from ground-truth segmentations,
this is done in three steps:
  1) Boundary perturbation (BSpline warp + directional field aligned to SDF gradient
     with observer-specific COM shifts),
  2) Edge-aware SDM noise (Sobel-derived image contrast uncertainty),
  3) Volume perturbation (SDF thresholding using observer-specific volume ratio).

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


# ---------------------------- Observer-specific parameters (from literature-reported values)--------------------------- #

OBSERVERS = [
    "Observer1",
    "Observer2",
    "Observer3",
    "Observer4",
    "Observer5",
    "Observer6",
]

VOLUME_RATIO_RANGES: Dict[str, Tuple[float, float]] = {
    "Observer1": (0.88, 1.46),
    "Observer2": (0.85, 1.17),
    "Observer3": (0.59, 1.16),
    "Observer4": (1.07, 1.55),
    "Observer5": (0.64, 1.12),
    "Observer6": (0.95, 1.08),
}


COM_VALUES: Dict[str, np.ndarray] = {
    "Observer1": np.array([2.0, 4.9, 4.0], dtype=np.float32),
    "Observer2": np.array([0.6, 2.2, 2.3], dtype=np.float32),
    "Observer3": np.array([5.9, 8.0, 6.5], dtype=np.float32),
    "Observer4": np.array([4.2, 3.2, 1.5], dtype=np.float32),
    "Observer5": np.array([1.2, 3.2, 5.0], dtype=np.float32),
    "Observer6": np.array([3.0, 2.5, 6.5], dtype=np.float32),
}


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
    target_volume = max(1, min(int(original_volume * target_ratio), sdf.size))

    flat = sdf.ravel()
    order = np.argsort(flat)[::-1]
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
    return 1.0 - normalize_volume(edge_np)


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
    return sdf + noise * (uncertainty_map.astype(np.float32) * float(base_std))


def directional_bias(
    seg_path: str,
    com_mean_shift_xyz: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Directional bias field aligned with unit SDF gradient.
    Bias magnitude is sampled per-axis around observer COM mean shift.
    """
    seg_img = sitk.ReadImage(seg_path)
    seg_np = (sitk.GetArrayFromImage(seg_img) > 0).astype(np.uint8)

    sdf = compute_signed_distance(seg_np)

    grad_z, grad_y, grad_x = np.gradient(sdf.astype(np.float32))
    gradient = np.stack([grad_x, grad_y, grad_z], axis=-1)

    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    unit_gradient = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm != 0)

    shape = unit_gradient.shape[:-1]
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
    size_xyz = reference_img.GetSize()
    control_points = [m + 3 for m in mesh_size]

    displacement_params: List[float] = []

    for z in np.linspace(0, size_xyz[2] - 1, control_points[2]):
        for y in np.linspace(0, size_xyz[1] - 1, control_points[1]):
            for x in np.linspace(0, size_xyz[0] - 1, control_points[0]):
                zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
                zi = max(0, min(zi, directional_bias_field.shape[0] - 1))
                yi = max(0, min(yi, directional_bias_field.shape[1] - 1))
                xi = max(0, min(xi, directional_bias_field.shape[2] - 1))
                disp_xyz = directional_bias_field[zi, yi, xi]
                displacement_params.extend(disp_xyz.tolist())

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
    transform = sitk.BSplineTransformInitializer(
        seg_img, [mesh_size] * seg_img.GetDimension(), order=3
    )

    bias_field = directional_bias(seg_path, com_mean_shift_xyz, rng=rng)
    transform = apply_directional_bias_to_BSpline(transform, bias_field, seg_img)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(seg_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)

    warped = resampler.Execute(seg_img)
    return (sitk.GetArrayFromImage(warped) > 0).astype(np.uint8)


def discover_cases(root_dir: str) -> List[CasePaths]:
    """Find cases under root_dir that contain *-t1c.nii.gz and *-seg.nii.gz."""
    cases: List[CasePaths] = []

    for folder in sorted(os.listdir(root_dir)):
        case_dir = os.path.join(root_dir, folder)
        if not os.path.isdir(case_dir):
            continue

        t1c_path, seg_path = None, None
        for f in os.listdir(case_dir):
            if f.endswith("-t1c.nii.gz"):
                t1c_path = os.path.join(case_dir, f)
            elif f.endswith("-seg.nii.gz"):
                seg_path = os.path.join(case_dir, f)

        if t1c_path and seg_path:
            case_id = os.path.basename(t1c_path).replace("-t1c.nii.gz", "")
            cases.append(CasePaths(case_id, t1c_path, seg_path))

    return cases


def save_mask(mask_zyx: np.ndarray, reference_seg_path: str, out_path: str) -> None:
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
    p.add_argument("--ground-truth-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mesh-size", type=int, default=8)
    p.add_argument("--base-std", type=float, default=0.5)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--numeric-observer-id", action="store_true")
    p.add_argument("--observers", nargs="*", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.verbose)

    os.makedirs(args.output_dir, exist_ok=True)
    cases = discover_cases(args.ground_truth_dir)

    if not cases:
        raise FileNotFoundError("No valid cases found in ground-truth directory.")

    observers = OBSERVERS if args.observers is None else args.observers
    rng = np.random.default_rng(args.seed)

    logging.info("Found %d cases.", len(cases))
    logging.info("Observers: %s", ", ".join(observers))

    n_written = 0

    for obs in observers:
        min_r, max_r = VOLUME_RATIO_RANGES[obs]
        com_shift = COM_VALUES[obs]

        for case in cases:
            warped = boundary_perturbations(
                case.t1c_path, case.seg_path, com_shift, rng, args.mesh_size
            )

            sdf = compute_signed_distance(warped)
            uncertainty = compute_edge_uncertainty_map(case.t1c_path)
            sdf = apply_sdm_perturbation_with_uncertainty(
                sdf, uncertainty, rng, args.base_std, args.smooth_sigma
            )

            mask, _ = threshold_sdf_masks(sdf, min_r, max_r, rng)
            if mask.sum() == 0:
                mask = warped

            obs_id = observer_output_id(obs, args.numeric_observer_id)
            out_path = os.path.join(
                args.output_dir, f"{case.case_id}_observer_{obs_id}.nii.gz"
            )
            save_mask(mask, case.seg_path, out_path)
            n_written += 1

    logging.warning("Done. Wrote %d masks to %s", n_written, args.output_dir)


if __name__ == "__main__":
    main()

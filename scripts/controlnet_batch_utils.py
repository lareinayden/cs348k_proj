"""Shared helpers for ControlNet batch scripts (Configs 3–6)."""

from pathlib import Path

import pandas as pd

DEFAULT_CFG_SCALES = [3.0, 5.0, 7.5, 10.0]
DEFAULT_CONTROLNET_SCALES = [0.5, 1.0, 1.5, 2.0]

CANNY_RUN_PATHS = {
    "empty": {
        "experiment_id": 3,
        "modality_name": "empty_text_plus_canny",
        "output_dir": "baseline_structural_canny",
        "result_csv": "baseline_structural_canny_results.csv",
        "scale_output_dir": "baseline_structural_canny",
        "scale_result_csv": "controlnet_scale_baseline_structural_canny_results.csv",
    },
    "dense": {
        "experiment_id": 5,
        "modality_name": "dense_text_plus_canny",
        "output_dir": "controlnet_canny",
        "result_csv": "controlnet_canny_results.csv",
        "scale_output_dir": "controlnet_canny",
        "scale_result_csv": "controlnet_scale_controlnet_canny_results.csv",
    },
}

SEG_RUN_PATHS = {
    "empty": {
        "experiment_id": 4,
        "modality_name": "empty_text_plus_seg",
        "output_dir": "baseline_structural_seg",
        "result_csv": "baseline_structural_seg_results.csv",
        "scale_output_dir": "baseline_structural_seg",
        "scale_result_csv": "controlnet_scale_baseline_structural_seg_results.csv",
    },
    "dense": {
        "experiment_id": 6,
        "modality_name": "dense_text_plus_seg",
        "output_dir": "controlnet_seg",
        "result_csv": "controlnet_seg_results.csv",
        "scale_output_dir": "controlnet_seg",
        "scale_result_csv": "controlnet_scale_controlnet_seg_results.csv",
    },
}


def resolve_run_paths(
    project_root: Path,
    modality: str,
    pipeline: str,
    controlnet_scale_sweep: bool = False,
    output_root: Path | None = None,
    result_csv: Path | None = None,
) -> tuple[Path, Path, int, str]:
    paths = CANNY_RUN_PATHS if pipeline == "canny" else SEG_RUN_PATHS
    cfg = paths[modality]

    if controlnet_scale_sweep:
        output_root_base = project_root / "outputs/controlNetScale" / cfg["scale_output_dir"]
        result_csv_path = project_root / "logs" / cfg["scale_result_csv"]
    else:
        output_root_base = project_root / "outputs" / cfg["output_dir"]
        result_csv_path = project_root / "logs" / cfg["result_csv"]

    if output_root is not None:
        output_root_base = output_root
    if result_csv is not None:
        result_csv_path = result_csv

    return output_root_base, result_csv_path, cfg["experiment_id"], cfg["modality_name"]


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def is_completed(
    results_df: pd.DataFrame,
    coco_id: int,
    guidance_scale: float,
    controlnet_scale: float,
) -> bool:
    if results_df.empty:
        return False
    mask = (
        (results_df["coco_id"] == coco_id)
        & (results_df["guidance_scale"] == guidance_scale)
        & (results_df["controlnet_conditioning_scale"] == controlnet_scale)
    )
    return bool(mask.any())


def append_result(results_df: pd.DataFrame, row: dict) -> pd.DataFrame:
    return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

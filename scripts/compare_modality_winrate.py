"""
Pairwise and multi-way win-rate comparison across experiment configurations.

For each image and metric, the best config earns 1 point (ties split 0.5 each).
Metrics: LPIPS ↓, DreamSim ↓, CLIP_Score ↑

Examples:
  python scripts/compare_modality_winrate.py --configs 2 3 4 --guidance-scale 7.5
  python scripts/compare_modality_winrate.py --configs 5 6 --guidance-scale 7.5
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_REGISTRY = {
    1: {
        "name": "sparse_text",
        "label": "Config 1 (sparse text)",
        "csv": PROJECT_ROOT / "logs/baseline_semantic_sparse_results.csv",
    },
    2: {
        "name": "dense_text",
        "label": "Config 2 (dense text)",
        "csv": PROJECT_ROOT / "logs/baseline_semantic_dense_results.csv",
    },
    3: {
        "name": "canny",
        "label": "Config 3 (empty + Canny)",
        "csv": PROJECT_ROOT / "logs/baseline_structural_canny_results.csv",
    },
    4: {
        "name": "seg",
        "label": "Config 4 (empty + seg mask)",
        "csv": PROJECT_ROOT / "logs/baseline_structural_seg_results.csv",
    },
    5: {
        "name": "dense_canny",
        "label": "Config 5 (dense + Canny)",
        "csv": PROJECT_ROOT / "logs/controlnet_canny_results.csv",
    },
    6: {
        "name": "dense_seg",
        "label": "Config 6 (dense + seg mask)",
        "csv": PROJECT_ROOT / "logs/controlnet_seg_results.csv",
    },
}

METRICS = {
    "LPIPS_Distance": "lower",
    "DreamSim_Distance": "lower",
    "CLIP_Score": "higher",
}

N_IMAGES = 40


def parse_args():
    parser = argparse.ArgumentParser(
        description="Win-rate comparison for 2 or 3 experiment configurations."
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        metavar="ID",
        help="Config IDs to compare (2–3 values, e.g. 2 3 4 or 5 6). Default: 2 3 4",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Per-image CSV path (default: logs/modality_comparison_<ids>_per_image.csv)",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Summary text path (default: logs/modality_comparison_<ids>_winrate.txt)",
    )
    return parser.parse_args()


def validate_configs(config_ids: list[int]) -> list[int]:
    if len(config_ids) < 2 or len(config_ids) > 3:
        raise ValueError(f"Provide 2 or 3 config IDs, got {len(config_ids)}: {config_ids}")

    unique = list(dict.fromkeys(config_ids))
    if len(unique) != len(config_ids):
        raise ValueError(f"Duplicate config IDs: {config_ids}")

    unknown = [c for c in config_ids if c not in CONFIG_REGISTRY]
    if unknown:
        valid = ", ".join(str(k) for k in sorted(CONFIG_REGISTRY))
        raise ValueError(f"Unknown config ID(s) {unknown}. Valid IDs: {valid}")

    return config_ids


def default_output_paths(config_ids: list[int]) -> tuple[Path, Path]:
    tag = "_".join(str(c) for c in config_ids)
    logs = PROJECT_ROOT / "logs"
    return (
        logs / f"modality_comparison_{tag}_per_image.csv",
        logs / f"modality_comparison_{tag}_winrate.txt",
    )


def load_config_df(cfg_id: int, guidance_scale: float, controlnet_scale: float) -> pd.DataFrame:
    path = CONFIG_REGISTRY[cfg_id]["csv"]
    if not path.exists():
        raise FileNotFoundError(f"Missing results for config {cfg_id}: {path}")

    df = pd.read_csv(path)
    if len(df[df["guidance_scale"] == guidance_scale]) == 0:
        available = sorted(df["guidance_scale"].unique())
        raise ValueError(
            f"Config {cfg_id}: no rows at CFG={guidance_scale}. "
            f"Available guidance scales in {path.name}: {available}"
        )

    df = df[df["guidance_scale"] == guidance_scale].copy()

    if "controlnet_conditioning_scale" in df.columns:
        df = df[df["controlnet_conditioning_scale"] == controlnet_scale]

    if len(df) != N_IMAGES:
        raise ValueError(
            f"Config {cfg_id}: expected {N_IMAGES} rows at CFG={guidance_scale}, got {len(df)}"
        )

    df = df.set_index("coco_id")
    return df


def score_winners(values: dict[int, float], direction: str) -> dict[int, float]:
    """Return fractional wins per config (1.0 winner, 0.5 each on tie)."""
    if direction == "lower":
        best = min(values.values())
        winners = [k for k, v in values.items() if v == best]
    else:
        best = max(values.values())
        winners = [k for k, v in values.items() if v == best]

    share = 1.0 / len(winners)
    return {cfg_id: share if cfg_id in winners else 0.0 for cfg_id in values}


def pairwise_wins(values: dict[int, float], direction: str, a: int, b: int) -> tuple[float, float, str]:
    """Return (points_a, points_b, outcome) for a head-to-head on one metric."""
    va, vb = values[a], values[b]
    if direction == "lower":
        if va < vb:
            return 1.0, 0.0, str(a)
        if vb < va:
            return 0.0, 1.0, str(b)
        return 0.5, 0.5, "tie"
    if va > vb:
        return 1.0, 0.0, str(a)
    if vb > va:
        return 0.0, 1.0, str(b)
    return 0.5, 0.5, "tie"


def format_config_header(config_ids: list[int], width: int = 8) -> str:
    return " ".join(f"{f'Cfg{c}':>{width}}" for c in config_ids)


def format_metric_row(
    metric: str,
    config_ids: list[int],
    points: dict[int, dict[str, float]],
    n_images: int,
) -> str:
    cols = " ".join(f"{points[c][metric] / n_images:>{7}.1%}" for c in config_ids)
    pts = "/".join(f"{points[c][metric]:.0f}" for c in config_ids)
    return f"{metric:<20} {cols}  ({pts} pts)"


def main():
    args = parse_args()
    config_ids = validate_configs(args.configs)

    if args.output_csv is None or args.output_summary is None:
        default_csv, default_summary = default_output_paths(config_ids)
        output_csv = args.output_csv or default_csv
        output_summary = args.output_summary or default_summary
    else:
        output_csv = args.output_csv
        output_summary = args.output_summary

    frames = {
        cfg_id: load_config_df(cfg_id, args.guidance_scale, args.controlnet_scale)
        for cfg_id in config_ids
    }

    common_ids = sorted(set.intersection(*(set(df.index) for df in frames.values())))
    if len(common_ids) != N_IMAGES:
        raise ValueError(f"Expected {N_IMAGES} shared coco_ids, found {len(common_ids)}")

    max_points_per_config = len(METRICS) * N_IMAGES
    config_labels = [CONFIG_REGISTRY[c]["label"] for c in config_ids]
    config_tag = ", ".join(str(c) for c in config_ids)

    per_image_rows = []
    multi_way_points = {cfg_id: {metric: 0.0 for metric in METRICS} for cfg_id in config_ids}
    pairwise_points = {
        pair: {cfg_id: {metric: 0.0 for metric in METRICS} for cfg_id in pair}
        for pair in combinations(config_ids, 2)
    }

    for coco_id in common_ids:
        row = {"coco_id": coco_id}
        for cfg_id, df in frames.items():
            for metric in METRICS:
                row[f"cfg{cfg_id}_{metric}"] = df.loc[coco_id, metric]

        for metric, direction in METRICS.items():
            values = {cfg_id: frames[cfg_id].loc[coco_id, metric] for cfg_id in config_ids}
            wins = score_winners(values, direction)

            for cfg_id, pts in wins.items():
                multi_way_points[cfg_id][metric] += pts

            winner_ids = [str(k) for k, v in wins.items() if v > 0]
            row[f"multi_way_winner_{metric}"] = "+".join(winner_ids) if len(winner_ids) > 1 else winner_ids[0]

            for a, b in combinations(config_ids, 2):
                pa, pb, outcome = pairwise_wins(values, direction, a, b)
                pairwise_points[(a, b)][a][metric] += pa
                pairwise_points[(a, b)][b][metric] += pb
                row[f"pair_{a}_vs_{b}_{metric}"] = outcome

        per_image_rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_image_rows).to_csv(output_csv, index=False)

    lines = []
    lines.append(f"Modality win-rate comparison: Configs {config_tag}")
    lines.append(
        f"CFG={args.guidance_scale}, ControlNet scale={args.controlnet_scale}, "
        f"n={N_IMAGES} images"
    )
    lines.append("")

    n_way_label = f"{len(config_ids)}-way"
    lines.append(
        f"{n_way_label} (best of {config_tag} per image per metric; ties split 0.5):"
    )
    lines.append(f"{'Metric':<20} {format_config_header(config_ids)}")
    for metric in METRICS:
        lines.append(format_metric_row(metric, config_ids, multi_way_points, N_IMAGES))

    total_multi = {cfg_id: sum(multi_way_points[cfg_id].values()) for cfg_id in config_ids}
    lines.append("")
    lines.append(
        f"{n_way_label} total points ({len(METRICS)} metrics × {N_IMAGES} images = "
        f"{max_points_per_config} max each):"
    )
    for cfg_id in config_ids:
        total = total_multi[cfg_id]
        lines.append(
            f"  {CONFIG_REGISTRY[cfg_id]['label']}: "
            f"{total:.1f} / {max_points_per_config} ({total / max_points_per_config:.1%})"
        )

    lines.append("")
    lines.append("Pairwise head-to-head win rates:")
    for (a, b), scores in sorted(pairwise_points.items()):
        lines.append(f"  Config {a} vs Config {b}:")
        for metric in METRICS:
            pa = scores[a][metric]
            pb = scores[b][metric]
            lines.append(
                f"    {metric:<20} cfg{a} {pa / N_IMAGES:>6.1%} ({pa:.0f}) | "
                f"cfg{b} {pb / N_IMAGES:>6.1%} ({pb:.0f})"
            )
        total_a = sum(scores[a].values())
        total_b = sum(scores[b].values())
        lines.append(
            f"    {'TOTAL (3 metrics)':<20} cfg{a} {total_a / max_points_per_config:>6.1%} "
            f"({total_a:.0f}) | cfg{b} {total_b / max_points_per_config:>6.1%} ({total_b:.0f})"
        )
        lines.append("")

    summary_text = "\n".join(lines)
    output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"\nConfigs: {', '.join(config_labels)}")
    print(f"Per-image breakdown: {output_csv}")
    print(f"Summary saved: {output_summary}")


if __name__ == "__main__":
    main()

"""Read dense vs. original captions from selected_coco_candidates.csv."""

DENSE_CAPTION_COL = "Dense Caption"


def get_dense_caption(row) -> str:
    """Enriched dense text for generation (Config 2/5/6) and CLIP ground truth."""
    if DENSE_CAPTION_COL in row.index:
        val = row[DENSE_CAPTION_COL]
        if val is not None and str(val).strip() and str(val) != "nan":
            return str(val).strip()
    return str(row["caption"]).strip()


def get_coco_caption(row) -> str:
    """Original COCO caption column (reference / logging)."""
    return str(row["caption"]).strip()

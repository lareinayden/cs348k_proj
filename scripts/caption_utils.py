"""Read dense vs. original captions from selected_coco_candidates CSV."""

DENSE_CAPTION_COLS = ("Dense Caption", "dense_caption")


def get_dense_caption(row) -> str:
    """Enriched dense text for generation (Config 2/5/6) and CLIP ground truth."""
    for col in DENSE_CAPTION_COLS:
        if col in row.index:
            val = row[col]
            if val is not None and str(val).strip() and str(val) != "nan":
                return str(val).strip()
    return str(row["caption"]).strip()


def get_coco_caption(row) -> str:
    """Original COCO caption column (reference / logging)."""
    return str(row["caption"]).strip()

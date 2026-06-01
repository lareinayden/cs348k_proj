# scripts/editing_tasks.py

EDIT_TASKS = {
    "cozy_indoor": {
        "sample_ids": [27620, 24610, 31248, 29596, 41872],
        "prompt_suffix": "cozy warm interior style, soft lighting, wooden textures, comfortable home atmosphere, inviting and realistic",
        "expected_better": "segmentation",
    },

    "cartoon_style": {
        "sample_ids": [52891, 39769, 12120, 100238, 60899],
        "prompt_suffix": "colorful cartoon illustration style, soft outlines, simplified shapes, playful colors, stylized but recognizable",
        "expected_better": "segmentation",
    },

    "photorealistic_layout": {
        "sample_ids": [12120, 27620, 2006, 98392, 25181],
        "prompt_suffix": "photorealistic image, natural lighting, detailed realistic objects, preserve the original scene layout",
        "expected_better": "canny",
    },

    "modern_redesign": {
        "sample_ids": [31248, 7574, 74209, 30213, 39484],
        "prompt_suffix": "modern minimalist redesign, clean surfaces, bright lighting, elegant contemporary style, same object layout",
        "expected_better": "segmentation",
    },
}
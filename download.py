# Download COCO dataset
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=100)
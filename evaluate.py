import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os

# === Configuration ===
# Update these paths accordingly
# ground_truth_path = '/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017_original.json'  # Path to COCO ground truth annotations
# predictions_path = '/home/emirhan/deteval/predictions_DINO.json'         # Path to your predictions.json file

ground_truth_path = 'class_agnostic_annotations.json'
predictions_path = 'class_agnostic_predictions.json'

img_dir = '/home/emirhan/datasets/object_detection/coco/val2017'
# =====================
# =====================

def main():
    # Initialize COCO ground truth
    print("Loading ground truth annotations...")
    coco_gt = COCO(ground_truth_path)

    # Load predictions
    print("Loading predictions...")
    with open(predictions_path, 'r') as f:
        predictions_data = json.load(f)
        predictions = predictions_data.get('predictions', [])

    if not predictions:
        print("No predictions found in the predictions.json file.")
        return

    # Convert predictions to COCO format
    print("Converting predictions to COCO format...")
    coco_dt = coco_gt.loadRes(predictions)

    # Initialize COCOeval for bbox
    print("Initializing COCO evaluation for bounding boxes...")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox', img_dir=img_dir)

    # Run evaluation
    print("Running evaluation...")
    coco_eval.params.useCats = 1  # Evaluate all categories
    coco_eval.params.imgIds = [139]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
if __name__ == '__main__':
    main()
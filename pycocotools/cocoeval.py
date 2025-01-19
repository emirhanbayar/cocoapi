__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', img_dir=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = []                  # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.img_dir = img_dir
        self.debug_vis = True
        self.vis_buffer = defaultdict(list)  # Buffer for visualization results

        # Import visualization libraries
        import cv2
        import os
        self.cv2 = cv2
        self.os = os

    def _visualize_debug(self, eval_imgs):
        """
        Create debug visualizations after evaluation, showing detections averaged across IoU range [0.50:0.95]
        """
        if not self.debug_vis or self.img_dir is None:
            return
            
        # Clear previous visualization buffer
        self.vis_buffer.clear()
        
        # Group eval results by image
        eval_by_img = defaultdict(list)
        for eval_img in eval_imgs:
            if eval_img is None:
                continue
            eval_by_img[eval_img['image_id']].append(eval_img)
        
        # Process each image
        for img_id, img_evals in eval_by_img.items():
            img_results = defaultdict(list)  # Store all detections/GTs for this image
            
            # Process each category's evaluation for this image
            for eval_img in img_evals:
                cat_id = eval_img['category_id']
                cat_name = self.cocoGt.cats[cat_id]['name'] if self.params.useCats else 'all'
                max_det = eval_img['maxDet']  # Get maxDet from evaluation results
                
                # Get corresponding detections and ground truths
                if self.params.useCats:
                    dts = self._dts[img_id, cat_id]
                    gts = self._gts[img_id, cat_id]
                else:
                    dts = [_ for cId in self.params.catIds for _ in self._dts[img_id, cId]]
                    gts = [_ for cId in self.params.catIds for _ in self._gts[img_id, cId]]

                # Skip if no detections or ground truths
                if len(dts) == 0 and len(gts) == 0:
                    continue

                # Sort detections by score and limit to maxDet
                if len(dts) > max_det:
                    dts = sorted(dts, key=lambda x: x['score'], reverse=True)[:max_det]

                dt_scores = eval_img['dtScores'][:max_det]  # Limit to maxDet
                dt_matches = eval_img['dtMatches'][:, :max_det]  # TxD matrix, limited to maxDet
                dt_ignore = eval_img['dtIgnore'][:, :max_det]    # TxD matrix, limited to maxDet
                gt_ignore = eval_img['gtIgnore']    # G vector
                
                # For each detection, count how many IoU thresholds it's a TP for
                T = len(self.params.iouThrs)
                tp_counts = np.zeros(len(dts))
                
                # for dt_idx in range(len(dts)):
                #     # Count IoU thresholds where this detection is a TP
                #     tp_count = 0
                #     for t in range(T):
                #         if not dt_ignore[t, dt_idx] and dt_matches[t, dt_idx] > 0:
                #             tp_count += 1
                #     tp_counts[dt_idx] = tp_count

                # # A detection is considered TP if it's TP for at least half of IoU thresholds
                # for dt_idx, (dt, score, tp_count) in enumerate(zip(dts, dt_scores, tp_counts)):
                #     is_tp = tp_count >= T/2  # TP if matched for majority of IoU thresholds
                #     if is_tp:
                #         img_results['detections'].append({
                #             'bbox': dt['bbox'],
                #             'score': score,
                #             'category': cat_name,
                #             'type': 'TP' if is_tp else 'FP'
                #         })
                
                # For ground truths, they're FN if they're unmatched at IoU ≥ 0.5
                gt_matches = eval_img['gtMatches'][0]  # Use IoU=0.5 threshold for FN
                for gt_idx, (gt, match, ignore) in enumerate(zip(gts, gt_matches, gt_ignore)):
                    if ignore:
                        continue
                        
                    if match == 0:  # Unmatched GT = False Negative
                        img_results['groundtruths'].append({
                            'bbox': gt['bbox'],
                            'category': cat_name,
                            'type': 'FN'
                        })
            
            # Draw visualization for this image
            if len(img_results['detections']) > 0 or len(img_results['groundtruths']) > 0:
                self._draw_image(img_id, img_results)

    def _visualize_matches(self, eval_imgs):
        """
        Create visualizations showing each TP detection with its matched ground truth.
        Each TP-GT pair gets its own image.
        """
        if not self.debug_vis or self.img_dir is None:
            return
            
        # Process each evaluation result
        for eval_img in eval_imgs:
            if eval_img is None:
                continue
                
            img_id = eval_img['image_id']
            cat_id = eval_img['category_id']
            
            # Get category name
            cat_name = self.cocoGt.cats[cat_id]['name'] if self.params.useCats else 'all'
            
            # Get corresponding detections and ground truths
            if self.params.useCats:
                dts = self._dts[img_id, cat_id]
                gts = self._gts[img_id, cat_id]
            else:
                dts = [_ for cId in self.params.catIds for _ in self._dts[img_id, cId]]
                gts = [_ for cId in self.params.catIds for _ in self._gts[img_id, cId]]
                
            max_det = eval_img['maxDet']
            if len(dts) > max_det:
                dts = sorted(dts, key=lambda x: x['score'], reverse=True)[:max_det]
                
            # Get evaluation matrices
            dt_matches = eval_img['dtMatches']  # TxD matrix
            dt_scores = eval_img['dtScores'][:max_det]  # D vector
            dt_ignore = eval_img['dtIgnore']    # TxD matrix
            gt_ignore = eval_img['gtIgnore']    # G vector

            T = len(self.params.iouThrs)  # Number of IoU thresholds
            
            # For each detection, check if it's a TP
            for dt_idx, (dt, score) in enumerate(zip(dts, dt_scores)):
                # Count IoU thresholds where this detection is a TP
                tp_count = 0
                matched_gt_ids = set()  # Store matched GT ids for this detection
                
                for t in range(T):
                    if not dt_ignore[t, dt_idx]:  # If detection should not be ignored
                        gt_match = dt_matches[t, dt_idx]
                        if gt_match > 0:  # If matched to a GT
                            tp_count += 1
                            matched_gt_ids.add(int(gt_match))
                
                # If detection is TP for majority of IoU thresholds
                if tp_count >= T/2:
                    # Load image
                    img_path = self.os.path.join(self.img_dir, f"{int(img_id):012d}.jpg")
                    if not self.os.path.exists(img_path):
                        print(f"Warning: Image {img_path} not found")
                        continue
                        
                    img = self.cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue

                    # For each GT this detection matched with
                    for gt_id in matched_gt_ids:
                        # Find the GT object
                        matched_gt = None
                        for gt in gts:
                            if gt['id'] == gt_id:
                                matched_gt = gt
                                break
                                
                        if matched_gt is None:
                            continue
                            
                        # Create a copy of the image for this match
                        img_match = img.copy()
                        
                        # Colors for visualization (BGR format)
                        dt_color = (0, 255, 0)  # Green for detection
                        gt_color = (255, 0, 0)  # Blue for ground truth
                        
                        # Draw detection
                        dt_bbox = [int(b) for b in dt['bbox']]
                        x, y, w, h = dt_bbox
                        self.cv2.rectangle(img_match, (x, y), (x + w, y + h), dt_color, 2)
                        dt_label = f"DT {cat_name} {score:.2f}"
                        self.cv2.putText(img_match, dt_label, (x, y - 5),
                                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, dt_color, 1)
                        
                        # Draw ground truth
                        gt_bbox = [int(b) for b in matched_gt['bbox']]
                        x, y, w, h = gt_bbox
                        self.cv2.rectangle(img_match, (x, y), (x + w, y + h), gt_color, 2)
                        gt_label = f"GT {cat_name}"
                        self.cv2.putText(img_match, gt_label, (x, y - 20),
                                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
                        
                        # Save visualization
                        save_dir = self.os.path.join('eval_vis', 'matches')
                        self.os.makedirs(save_dir, exist_ok=True)
                        save_path = self.os.path.join(save_dir, 
                            f"{int(img_id):012d}_cat{cat_id}_dt{dt['id']}_gt{gt_id}.jpg")
                        self.cv2.imwrite(save_path, img_match)

    def analyze_matches(self, eval_imgs):
        """
        Analyze cases where a single detection matches multiple ground truths
        when useCats=1 but can't when useCats=0
        """
        dt_to_gts = defaultdict(list)  # Detection -> list of matched GTs
        multi_matches = []  # Store cases where one detection matches multiple GTs
        
        # Process each evaluation result
        for eval_img in eval_imgs:
            if eval_img is None:
                continue
                
            img_id = eval_img['image_id']
            cat_id = eval_img['category_id']
            
            # Get detections and ground truths
            if self.params.useCats:
                dts = self._dts[img_id, cat_id]
                gts = self._gts[img_id, cat_id]
            else:
                dts = [_ for cId in self.params.catIds for _ in self._dts[img_id, cId]]
                gts = [_ for cId in self.params.catIds for _ in self._gts[img_id, cId]]
                
            max_det = eval_img['maxDet']
            if len(dts) > max_det:
                dts = sorted(dts, key=lambda x: x['score'], reverse=True)[:max_det]
                
            dt_matches = eval_img['dtMatches']  # TxD matrix
            dt_scores = eval_img['dtScores'][:max_det]
            dt_ignore = eval_img['dtIgnore']
            gt_ignore = eval_img['gtIgnore']
            
            T = len(self.params.iouThrs)
            
            # For each detection
            for dt_idx, (dt, score) in enumerate(zip(dts, dt_scores)):
                matched_gts = set()
                
                # Check matches across IoU thresholds
                for t in range(T):
                    if not dt_ignore[t, dt_idx]:
                        gt_match = dt_matches[t, dt_idx]
                        if gt_match > 0:
                            matched_gts.add(int(gt_match))
                
                # If this detection matched multiple GTs
                if len(matched_gts) > 1:
                    matched_gt_info = []
                    # Get IoUs with all matched GTs
                    for gt_id in matched_gts:
                        matched_gt = None
                        for gt in gts:
                            if gt['id'] == gt_id:
                                matched_gt = gt
                                break
                        
                        if matched_gt:
                            # Compute IoU between detection and GT
                            dt_bbox = dt['bbox']
                            gt_bbox = matched_gt['bbox']
                            
                            # Simple IoU computation for bboxes
                            def compute_iou(bbox1, bbox2):
                                x1, y1, w1, h1 = bbox1
                                x2, y2, w2, h2 = bbox2
                                
                                # Convert to x1,y1,x2,y2 format
                                box1 = [x1, y1, x1+w1, y1+h1]
                                box2 = [x2, y2, x2+w2, y2+h2]
                                
                                # Intersection
                                xi1 = max(box1[0], box2[0])
                                yi1 = max(box1[1], box2[1])
                                xi2 = min(box1[2], box2[2])
                                yi2 = min(box1[3], box2[3])
                                
                                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                                
                                # Union
                                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                                union_area = box1_area + box2_area - inter_area
                                
                                return inter_area / union_area if union_area > 0 else 0
                            
                            iou = compute_iou(dt_bbox, gt_bbox)
                            
                            matched_gt_info.append({
                                'gt_id': gt_id,
                                'category_id': matched_gt.get('category_id'),
                                'iou': iou
                            })
                    
                    if matched_gt_info:
                        multi_matches.append({
                            'img_id': img_id,
                            'dt_id': dt['id'],
                            'dt_score': score,
                            'dt_category': dt.get('category_id'),
                            'matches': matched_gt_info
                        })
        
        # Print analysis
        if multi_matches:
            print(f"\nFound {len(multi_matches)} cases where a detection matches multiple ground truths:")
            for case in multi_matches:
                print(f"\nImage {case['img_id']}, Detection {case['dt_id']} (score: {case['dt_score']:.3f}, category: {case['dt_category']}):")
                for match in case['matches']:
                    print(f"  Matched GT {match['gt_id']} (category: {match['category_id']}) with IoU {match['iou']:.3f}")
        else:
            print("\nNo cases found where a detection matches multiple ground truths")

        return multi_matches
    
    def _visualize_multi_matches(self, eval_imgs):
        """
        Create visualizations highlighting detections that match multiple ground truths.
        Colors:
        - Regular TPs: Green
        - Regular GTs: Blue
        - Multi-match detection: Yellow
        - GTs matched by multi-match detection: Purple
        """
        if not self.debug_vis or self.img_dir is None:
            return
            
        # First run the match analysis to find multi-match cases
        multi_matches = self.analyze_matches(eval_imgs)
        
        # Group multi-matches by image
        multi_match_by_img = defaultdict(list)
        for match in multi_matches:
            multi_match_by_img[match['img_id']].append(match)
        
        # Process each evaluation result
        for eval_img in eval_imgs:
            if eval_img is None:
                continue
                
            img_id = eval_img['image_id']
            cat_id = eval_img['category_id']
            
            # Skip if no multi-matches for this image
            if img_id not in multi_match_by_img:
                continue
                
            # Get detections and ground truths
            if self.params.useCats:
                dts = self._dts[img_id, cat_id]
                gts = self._gts[img_id, cat_id]
            else:
                dts = [_ for cId in self.params.catIds for _ in self._dts[img_id, cId]]
                gts = [_ for cId in self.params.catIds for _ in self._gts[img_id, cId]]
                
            max_det = eval_img['maxDet']
            if len(dts) > max_det:
                dts = sorted(dts, key=lambda x: x['score'], reverse=True)[:max_det]
                
            dt_matches = eval_img['dtMatches']
            dt_scores = eval_img['dtScores'][:max_det]
            dt_ignore = eval_img['dtIgnore']
            gt_ignore = eval_img['gtIgnore']
            
            # Load image
            img_path = self.os.path.join(self.img_dir, f"{int(img_id):012d}.jpg")
            if not self.os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found")
                continue
                
            img = self.cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Colors (BGR format)
            colors = {
                'tp': (0, 255, 0),      # Green
                'gt': (255, 0, 0),      # Blue
                'multi': (0, 255, 255),  # Yellow
                'matched': (255, 0, 255) # Purple
            }
            
            # Get multi-match detections and their matched GTs for this image
            multi_match_dts = {m['dt_id']: m for m in multi_match_by_img[img_id]}
            multi_match_gts = set()
            for match in multi_match_by_img[img_id]:
                for gt_match in match['matches']:
                    multi_match_gts.add(gt_match['gt_id'])
            
            # Draw all detections and ground truths
            font = self.cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # First draw regular TPs and GTs
            T = len(self.params.iouThrs)
            for dt_idx, (dt, score) in enumerate(zip(dts, dt_scores)):
                if dt['id'] in multi_match_dts:
                    continue  # Skip multi-match detections for now
                    
                # Check if it's a TP
                tp_count = 0
                for t in range(T):
                    if not dt_ignore[t, dt_idx] and dt_matches[t, dt_idx] > 0:
                        tp_count += 1
                
                if tp_count >= T/2:  # If TP for majority of thresholds
                    bbox = [int(b) for b in dt['bbox']]
                    x, y, w, h = bbox
                    self.cv2.rectangle(img, (x, y), (x + w, y + h), colors['tp'], thickness)
                    label = f"TP {score:.2f}"
                    self.cv2.putText(img, label, (x, y - 5), font, font_scale, colors['tp'], thickness)
            
            # Draw ground truths that aren't part of multi-matches
            for gt in gts:
                if gt['id'] in multi_match_gts:
                    continue  # Skip multi-match GTs for now
                    
                if not gt['ignore']:
                    bbox = [int(b) for b in gt['bbox']]
                    x, y, w, h = bbox
                    self.cv2.rectangle(img, (x, y), (x + w, y + h), colors['gt'], thickness)
                    label = "GT"
                    self.cv2.putText(img, label, (x, y - 5), font, font_scale, colors['gt'], thickness)
            
            # Now draw multi-match detections and their matched GTs
            for match in multi_match_by_img[img_id]:
                # Draw the detection in yellow
                dt = None
                for d in dts:
                    if d['id'] == match['dt_id']:
                        dt = d
                        break
                        
                if dt:
                    bbox = [int(b) for b in dt['bbox']]
                    x, y, w, h = bbox
                    self.cv2.rectangle(img, (x, y), (x + w, y + h), colors['multi'], thickness)
                    label = f"Multi {match['dt_score']:.2f}"
                    self.cv2.putText(img, label, (x, y - 5), font, font_scale, colors['multi'], thickness)
                    
                    # Draw matched GTs in purple
                    for gt_match in match['matches']:
                        for gt in gts:
                            if gt['id'] == gt_match['gt_id']:
                                bbox = [int(b) for b in gt['bbox']]
                                x, y, w, h = bbox
                                self.cv2.rectangle(img, (x, y), (x + w, y + h), colors['matched'], thickness)
                                label = f"Matched GT ({gt_match['iou']:.2f})"
                                self.cv2.putText(img, label, (x, y - 5), font, font_scale, colors['matched'], thickness)
            
            # Save visualization
            save_dir = self.os.path.join('eval_vis', 'multi_matches')
            self.os.makedirs(save_dir, exist_ok=True)
            save_path = self.os.path.join(save_dir, f"{int(img_id):012d}_multi_match.jpg")
            self.cv2.imwrite(save_path, img)

    def _draw_image(self, img_id, results):
        """Helper method to draw detections on an image"""
        # Load image
        img_path = self.os.path.join(self.img_dir, f"{int(img_id):012d}.jpg")
        if not self.os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            return
            
        img = self.cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return

        # Colors for visualization (BGR format)
        colors = {
            'TP': (0, 255, 0),    # Green
            'FP': (0, 0, 255),    # Red
            'FN': (255, 0, 0)     # Blue
        }
        
        thickness = 1
        font = self.cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        # Draw all results for this image
        all_results = results['detections'] + results['groundtruths']
        for result in all_results:
            bbox = result['bbox']
            x, y, w, h = [int(b) for b in bbox]
            
            color = colors[result['type']]
            
            # Draw bbox
            self.cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare label
            if result['type'] in ['TP', 'FP']:
                label = f"{result['type']} {result['category']} {result['score']:.2f}"
            else:  # FN
                label = f"{result['type']} {result['category']}"
                
            # Draw label with background for better visibility
            (label_w, label_h), baseline = self.cv2.getTextSize(label, font, font_scale, thickness)
            # self.cv2.rectangle(img, (x, y - label_h - baseline - 5), (x + label_w, y), color, -1)
            self.cv2.putText(img, label, (x, y - baseline - 5),
                        font, font_scale, color, thickness)

        # Save visualization
        save_dir = self.os.path.join('eval_vis')
        self.os.makedirs(save_dir, exist_ok=True)
        save_path = self.os.path.join(save_dir, f"{int(img_id):012d}_debug.jpg")
        self.cv2.imwrite(save_path, img)

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]

        self._paramsEval = copy.deepcopy(self.params)

        # Create debug visualizations after evaluation is complete
        if self.debug_vis:
            self._visualize_debug(self.evalImgs)
            # self._visualize_matches(self.evalImgs)
            self._visualize_multi_matches(self.evalImgs)
            self.analyze_matches(self.evalImgs)
        
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((13,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[3])
            stats[10] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100, 300]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

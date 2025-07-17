# MIT License
#
# Copyright (c) 2025 Luca Lobefaro, Matteo Sodano, Tiziano Guadagnino, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


def get_max_association(label: int, occurrances_counter: dict, scale_factor: int):
    max_key = -1
    max_n_occurrances = 0
    for key in occurrances_counter:
        if key % scale_factor == label:
            if occurrances_counter[key] > max_n_occurrances:
                max_key = key // scale_factor
                max_n_occurrances = occurrances_counter[key]
    return max_key, max_n_occurrances


def run_semantic_evaluation(
    dataset,
    predicted_masks: List[np.ndarray],
    scale_factor: int = 1000,
    visualize: bool = False,
) -> Tuple[Dict, Dict]:
    # Initialize metrics values for detection
    tp_detection = 0
    fn_detection = 0
    fp_detection = 0

    # Initialize metrics values for matching
    tp_association = 0
    fn_association = 0

    # Initialize other stuff
    gt2pred_current_frame = {}
    gt2pred_associations = {}

    # For each frame
    for image_idx, prediction in enumerate(predicted_masks):
        # Clean current frame informations
        gt2pred_current_frame.clear()

        # Take the corresponing gt
        ground_truth_mask = dataset.get_semantic_mask(image_idx)

        # Extract our predicted labels
        prediction_labels = np.unique(prediction)

        # Magic trick to have gt labels on another order of number
        ground_truth_mask = ground_truth_mask * scale_factor
        mask_union = ground_truth_mask + prediction
        n_gt_labels = len(np.unique(ground_truth_mask)) - 1

        # Associate each predicted label to a ground truth label
        for label in prediction_labels:

            if label == 0:  # Ignore background label
                continue

            # Get the gt label that correspond to this instance
            # (the one that better overlap in the gt mask)
            unique, counts = np.unique(mask_union, return_counts=True)
            occurrances_counter = dict(zip(unique, counts))

            # Background pixels (in gt) predicted as fruit
            n_pixels_not_associated = (
                occurrances_counter[label] if label in occurrances_counter else 0
            )

            # Get best matched gt label and check that overlaps enough
            matched_gt_label, n_pixel_overlapping = get_max_association(
                label, occurrances_counter, scale_factor
            )
            if matched_gt_label == 0 or n_pixel_overlapping < n_pixels_not_associated:
                # Case in which we allucinate fruits
                fp_detection += 1
                continue

            # If it is the first time we associate to this gt value
            if matched_gt_label not in gt2pred_current_frame:
                gt2pred_current_frame[matched_gt_label] = (label, n_pixel_overlapping)
            else:
                # Check if the current prediction better overlap the gt than the previous one
                _, previous_n_pixel_overlapping = gt2pred_current_frame[
                    matched_gt_label
                ]
                if n_pixel_overlapping > previous_n_pixel_overlapping:
                    gt2pred_current_frame[matched_gt_label] = (
                        label,
                        n_pixel_overlapping,
                    )

        # Finish evaluation on detection
        tp_detection += len(gt2pred_current_frame.keys())
        fn_detection += n_gt_labels - len(gt2pred_current_frame.keys())

        # Evaluate associations
        for gt_label, value in gt2pred_current_frame.items():
            pred_label, _ = value
            if gt_label in gt2pred_associations:
                if gt2pred_associations[gt_label][-1] == pred_label:
                    tp_association += 1
                else:
                    fn_association += 1
                # Update association for tracking consistency
                gt2pred_associations[gt_label].append(pred_label)
            else:
                gt2pred_associations[gt_label] = [pred_label]

    # Compute other metrics
    metrics = {}
    metrics["detection_recall"] = tp_detection / (tp_detection + fn_detection)
    metrics["detection_accuracy"] = tp_detection / (
        tp_detection + fp_detection + fn_detection
    )
    metrics["detection_precision"] = tp_detection / (tp_detection + fp_detection)
    metrics["detection_f1_score"] = (
        2 * tp_detection / (2 * tp_detection + fp_detection + fn_detection)
    )
    metrics["associations_recall"] = tp_association / (tp_association + fn_association)
    metrics["associations_f1_score"] = (
        2 * tp_association / (2 * tp_association + fn_association)
    )
    metrics["tp_association"] = tp_association
    metrics["fn_association"] = fn_association

    # Take only the max occurences for labels matches gt <-> pred
    gt2pred_associations_max = {}
    for key, value in gt2pred_associations.items():
        gt2pred_associations_max[int(key)] = int(max(value, key=value.count))

    return metrics, gt2pred_associations_max


def run_semantic_evaluation_intersequences(
    dataset,
    predicted_masks: List[np.ndarray],
    reference_gt2pred_associations: Dict,
    scale_factor: int = 1000,
    visualize: bool = False,
) -> Dict:
    # Initialize metrics values for detection
    tp_detection = 0
    fn_detection = 0
    fp_detection = 0

    # Initialize metrics values for matching
    tp_association = 0
    fn_association = 0

    # Initialize other stuff
    gt2pred_current_frame = {}
    gt2pred_associations = {}

    # For each frame
    for image_idx, prediction in enumerate(predicted_masks):
        # Clean current frame informations
        gt2pred_current_frame.clear()

        # Take the corresponing gt
        ground_truth_mask = dataset.get_semantic_mask(image_idx)

        # Extract our predicted labels
        prediction_labels = np.unique(prediction)

        # Magic trick to have gt labels on another order of number
        ground_truth_mask = ground_truth_mask * scale_factor
        mask_union = ground_truth_mask + prediction
        n_gt_labels = len(np.unique(ground_truth_mask)) - 1

        # Associate each predicted label to a ground truth label
        for label in prediction_labels:

            if label == 0:  # Ignore background label
                continue

            # Get the gt label that correspond to this instance
            # (the one that better overlap in the gt mask)
            unique, counts = np.unique(mask_union, return_counts=True)
            occurrances_counter = dict(zip(unique, counts))

            # Background pixels (in gt) predicted as fruit
            n_pixels_not_associated = (
                occurrances_counter[label] if label in occurrances_counter else 0
            )

            # Get best matched gt label and check that overlaps enough
            matched_gt_label, n_pixel_overlapping = get_max_association(
                label, occurrances_counter, scale_factor
            )
            if n_pixel_overlapping < n_pixels_not_associated:
                # Case in which we allucinate fruits
                fp_detection += 1
                continue

            # If it is the first time we associate to this gt value
            if matched_gt_label not in gt2pred_current_frame:
                gt2pred_current_frame[matched_gt_label] = (label, n_pixel_overlapping)
            else:
                # Check if the current prediction better overlap the gt than the previous one
                _, previous_n_pixel_overlapping = gt2pred_current_frame[
                    matched_gt_label
                ]
                if n_pixel_overlapping > previous_n_pixel_overlapping:
                    gt2pred_current_frame[matched_gt_label] = (
                        label,
                        n_pixel_overlapping,
                    )

        # Finish evaluation on detection
        tp_detection += len(gt2pred_current_frame.keys())
        fn_detection += n_gt_labels - len(gt2pred_current_frame.keys())

        # Evaluate associations
        for gt_label, value in gt2pred_current_frame.items():
            pred_label, _ = value
            if gt_label in reference_gt2pred_associations:
                if pred_label == reference_gt2pred_associations[gt_label]:
                    tp_association += 1
                else:
                    fn_association += 1

            if gt_label in gt2pred_associations.keys():
                gt2pred_associations[gt_label].append(pred_label)
            else:
                gt2pred_associations[gt_label] = [pred_label]

    # Compute other metrics
    metrics = {}
    metrics["detection_recall"] = tp_detection / (tp_detection + fn_detection)
    metrics["detection_accuracy"] = tp_detection / (
        tp_detection + fp_detection + fn_detection
    )
    metrics["detection_precision"] = tp_detection / (tp_detection + fp_detection)
    metrics["detection_f1_score"] = (
        2 * tp_detection / (2 * tp_detection + fp_detection + fn_detection)
    )
    metrics["associations_recall"] = tp_association / (tp_association + fn_association)
    metrics["associations_f1_score"] = (
        2 * tp_association / (2 * tp_association + fn_association)
    )
    metrics["tp_association"] = tp_association
    metrics["fn_association"] = fn_association

    return metrics


def get_centers_and_radius(points, labels):
    centers = []
    radiuses = []
    centers_ids = []

    ids = np.unique(labels)
    for id in ids:
        # Ignore background
        if id == 0:
            continue

        # Take only points with given label
        pts_u = []
        for pt, label in zip(points, labels):
            if label == id:
                pts_u.append(pt)
        pts_u = np.asarray(pts_u)

        center = pts_u.mean(axis=0)
        assert center.shape[0] == 3, "something wrong with np mean axis"
        radius = np.linalg.norm(pts_u - center, axis=1, ord=2).max()
        if radius > 0:
            centers.append(center)
            radiuses.append(radius)
            centers_ids.append(id)

    return centers, radiuses, centers_ids


def get_sphere_volume(r):
    return 4 / 3 * np.pi * (r**3)


def get_intersection_volume(c0, c1, r0, r1):
    d = np.linalg.norm(c0 - c1)
    if d >= (r0 + r1):
        return 0
    if d <= abs(r0 - r1):
        return get_sphere_volume(min(r0, r1))
    return (
        np.pi
        * (r0 + r1 - d) ** 2
        * (d**2 + 2 * d * (r0 + r1) - 3 * (r0 - r1) ** 2)
        / 12.0
        / d
    )


def get_iou(c0, c1, r0, r1):
    intersection = get_intersection_volume(c0, c1, r0, r1)
    union = get_sphere_volume(r0) + get_sphere_volume(r1) - intersection
    return intersection / union


def associate_pred_gt(
    centers_pred, centers_gt, radiuses_pred, radiuses_gt, ids_pred, ids_gt, iou_th
):
    pred2gt = {}
    for center_gt, radius_gt, id_gt in zip(centers_gt, radiuses_gt, ids_gt):

        ious = []
        for center_pred, radius_pred in zip(centers_pred, radiuses_pred):
            iou = get_iou(center_gt, center_pred, radius_gt, radius_pred)
            ious.append(iou)
        ious = np.asarray(ious)

        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]
        best_pred_id = ids_pred[best_idx]

        if best_iou > iou_th:
            pred2gt[best_pred_id] = id_gt

    return pred2gt


def pred2gt_associations(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    labels_pred: List[int],
    labels_gt: List[int],
    iou_th: float,
):
    centers_pred, radiuses_pred, ids_pred = get_centers_and_radius(
        points_pred, labels_pred
    )
    centers_gt, radiuses_gt, ids_gt = get_centers_and_radius(points_gt, labels_gt)

    # Associate pred to gt using IoU
    pred2gt = associate_pred_gt(
        centers_pred, centers_gt, radiuses_pred, radiuses_gt, ids_pred, ids_gt, iou_th
    )

    return pred2gt


def evaluate_map_associations(pred2gt, ref_pred2gt):
    # Initialization (tpm: true_positive_matches, wpm: wrong_positive_matches)
    tpm, wpm, tn, fp, fn = 0, 0, 0, 0, 0
    precision, recall, f1 = 0, 0, 0
    ref_pred_ids = ref_pred2gt.keys()
    ref_gt_ids = ref_pred2gt.values()

    for pred_label, gt_label in pred2gt.items():
        if pred_label not in ref_pred_ids:
            if gt_label not in ref_gt_ids:
                tn += 1
            else:
                fn += 1
        else:
            if gt_label == ref_pred2gt[pred_label]:
                tpm += 1
            else:
                if gt_label in ref_gt_ids:
                    wpm += 1
                else:
                    fp += 1

    # tot = pred.shape[0]            ## aka
    tot = tpm + wpm + tn + fp + fn  ## just for fun

    accuracy = (tpm + tn) / tot

    if (tpm + fp + wpm) > 0:
        precision = tpm / (tpm + fp + wpm)

    if (tpm + fn) > 0:
        recall = tpm / (tpm + fn)

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "acc": accuracy,
        "prec": precision,
        "rec": recall,
        "f1": f1,
        "tot": tot,
        "tpm": tpm,
        "wpm": wpm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "cor": tn + tpm,
        "z": tn + fn,
        "gtz": tn + fp,
    }

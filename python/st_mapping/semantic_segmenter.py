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
import numpy as np
from typing import List, Tuple
from st_mapping.config.config import StMappingConfig
from st_mapping.superpoint.image_matcher import ImageMatcher
from st_mapping.core.mapping import unproject_image_point
from ultralytics import YOLO
import cv2
import copy
import math


class SemanticSegmenter:

    # --- PUBLIC INTERFACE ---------------------------------------------------------------
    def __init__(
        self,
        dataset,
        config: StMappingConfig,
        odometry,
        ref_dataset=None,
        ref_instances_centers: List[np.ndarray] = [],
    ):
        self._camera_matrix_inv = np.linalg.inv(dataset.get_intrinsic())
        self._camera_extrinsics = dataset.get_extrinsics()

        self._dataset = dataset
        self._ref_dataset = ref_dataset
        self._config = config
        self._odometry = odometry
        self._instances_centers = ref_instances_centers
        self._last_ref_center_idx = len(self._instances_centers) - 1
        self._matched_ref_centers = [False] * len(self._instances_centers)

        # For 2d matching
        self._2d_centers = []
        self._bounding_boxes = []
        self._image_ids = []
        self._current_image_2d_centers = []
        self._current_image_bounding_boxes = []
        self._current_image_ids = []
        self._ref_img: np.ndarray = np.asarray([])

        self._yolo = YOLO(self._config.semantic.yolo_weights_path)
        self._image_matcher = ImageMatcher(config)

    def extract_semantics(self, rgb_img, depth_img):
        # Guess a pose for the current frame
        robot_pose = self._odometry.guess_next_pose()
        camera_pose = robot_pose @ self._camera_extrinsics

        # Inizialize the centers and bounding boxes
        reference_centers = []
        reference_bounding_boxes = []
        reference_ids = []
        if len(self._2d_centers) > 0:
            reference_centers = self._2d_centers[-1]
            reference_bounding_boxes = self._bounding_boxes[-1]
            reference_ids = self._image_ids[-1]

        # Get the reference image and use their centers2d
        # othrwise the loader will complain if the centers2d where not computed
        if self._ref_dataset is not None:
            ref_rgb_img, _, _, ref_frame_idx = (
                self._ref_dataset.get_nearest_image_and_pose(robot_pose)
            )
            ref_centers2d, ref_bounding_boxes, ref_ids = self._ref_dataset.load_boxes()
            reference_centers = ref_centers2d[ref_frame_idx]
            reference_bounding_boxes = ref_bounding_boxes[ref_frame_idx]
            reference_ids = ref_ids[ref_frame_idx]
            self._ref_img = ref_rgb_img

        # Image matching
        kps = np.asarray([])
        ref_kps = np.asarray([])
        if len(self._ref_img) != 0:
            kps, ref_kps = self._match_images(rgb_img, self._ref_img)

        # Instance segmentation
        yolo_results = self._yolo(rgb_img, conf=0.5, verbose=False)

        # Get image to visualize
        labeled_rgb_img = yolo_results[0].plot(labels=False, conf=False)

        # Compute mask and instance centers
        instances_mask = np.zeros(depth_img.shape[:2], np.uint64)
        if yolo_results[0].masks is None:
            return labeled_rgb_img, instances_mask
        for i in range(0, len(yolo_results[0].masks.xy)):
            # Compute instance mask
            binary_mask = np.zeros(depth_img.shape[:2], np.uint8)
            contour = yolo_results[0].masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
            if not len(contour) > 0:
                continue
            cv2.drawContours(binary_mask, [contour], -1, (1, 0, 0), cv2.FILLED)

            # Process only instances inside the depth threshold
            average_depth_val = np.nanmean(depth_img[binary_mask == 1])
            if (
                average_depth_val > self._config.dataset.depth_min_th
                and average_depth_val < self._config.dataset.depth_max_th
            ):

                # Compute bounding box and convert it to x/y min, x/y max values
                bbx = yolo_results[0].boxes[i].xywh[0].cpu().numpy().astype(int)
                bounding_box = np.hstack(
                    [
                        bbx[:2] - bbx[2:] / 2,
                        bbx[:2] + bbx[2:] / 2,
                    ]
                )

                # Get bounding box center depth value
                box_center_2d = bbx[:2]
                box_center_depth = depth_img[box_center_2d[1], box_center_2d[0]]

                # Filter out invalid center
                if (
                    math.isnan(box_center_depth)
                    or box_center_depth <= self._config.dataset.depth_min_th
                    or box_center_depth >= self._config.dataset.depth_max_th
                ):
                    continue

                # Compute the box center in 3D
                box_center_3d = unproject_image_point(
                    box_center_2d[0],
                    box_center_2d[1],
                    box_center_depth,
                    self._camera_matrix_inv,
                    camera_pose,
                )

                # Compute the instance number
                if self._config.semantic.method == "centers3d":
                    instance_number = self._compute_instance_number_centers3d(
                        box_center_3d
                    )
                elif self._config.semantic.method == "centers2d":
                    instance_number = self._compute_instance_number_centers2d(
                        box_center_2d,
                        reference_centers,
                        reference_ids,
                        box_center_3d,
                    )
                elif self._config.semantic.method == "IoU":
                    instance_number = self._compute_instance_number_IoU(
                        bounding_box,
                        reference_bounding_boxes,
                        reference_ids,
                        box_center_3d,
                    )
                elif self._config.semantic.method == "image_matches":
                    instance_number = self._compute_instance_keypoints_matches(
                        bounding_box,
                        kps,
                        ref_kps,
                        reference_bounding_boxes,
                        reference_ids,
                        box_center_3d,
                    )
                else:
                    print("[ERROR]: Invalid method choosed")
                    exit(1)

                # Udate the instance mask
                instances_mask = instances_mask + (binary_mask * instance_number)

        # Save infos for next frame
        self._save_infos_for_next_frame(rgb_img)

        return labeled_rgb_img, instances_mask

    def get_dead_fruits_ids(self):
        return np.where(~np.asarray(self._matched_ref_centers))[0] + 1

    def get_born_fruits_ids(self):
        return (
            np.arange(self._last_ref_center_idx + 1, len(self._instances_centers)) + 1
        )

    def get_all_instances_centers(self) -> np.ndarray:
        return np.asarray(self._instances_centers)

    def get_ref_instances_centers(self) -> np.ndarray:
        return np.asarray(self._instances_centers[: self._last_ref_center_idx + 1])

    # Return the indices of the centers that we've already seen in the ref
    def get_matched_instances_indices(self) -> np.ndarray:
        if self._last_ref_center_idx == -1:
            return np.asarray([])
        return np.where(np.asarray(self._matched_ref_centers))[0]

    # Returns the centers for instances never seen in the ref
    def get_new_instances_centers(self) -> np.ndarray:
        if self._last_ref_center_idx == -1:
            return self.get_all_instances_centers()
        return np.asarray(self._instances_centers[self._last_ref_center_idx + 1 :])

    def get_boxes(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        return self._2d_centers, self._bounding_boxes, self._image_ids

    # --- PRIVATE INTERFACE ---------------------------------------------------------------
    # CONSIDERAZIONE: il problema di questo metodo e' che in qualche immagine qualche istanza
    # potrebbe non essere visibile e quando torna visibile nell'immagine successiva viene
    # considerato un nuovo centro, una possibile soluzione e' quella di usare una finestra
    def _compute_instance_number_centers2d(
        self,
        box_center_2d: np.ndarray,
        reference_centers,
        reference_ids,
        box_center_3d: np.ndarray,
    ):
        # Save the 2D center
        self._current_image_2d_centers.append(box_center_2d)

        # If this is not the first image
        if len(reference_centers) != 0:
            # Find the nearest center in the previous image
            distances = np.sum((reference_centers - box_center_2d) ** 2, axis=1)
            min_distance_idx = np.argmin(distances).astype(int)
            if distances[min_distance_idx] <= 300:
                instance_number = reference_ids[min_distance_idx]
                self._instances_centers[instance_number - 1] = box_center_3d
                if instance_number - 1 <= self._last_ref_center_idx:
                    self._matched_ref_centers[instance_number - 1] = True
                self._current_image_ids.append(instance_number)
                return instance_number

        self._instances_centers.append(box_center_3d)
        instance_number = len(self._instances_centers)
        self._current_image_ids.append(instance_number)
        return instance_number

    # CONSIDERAZIONE: stessa di _compute_instance_number_centers2d
    def _compute_instance_number_IoU(
        self,
        bounding_box,
        reference_bounding_boxes,
        reference_ids,
        box_center_3d: np.ndarray,
    ):
        # Save the bounding box
        self._current_image_bounding_boxes.append(bounding_box)

        # If this is not the first image
        if len(reference_bounding_boxes) != 0:
            # Find the most overlapping bounding box in the previous image
            ious = np.array(
                [
                    SemanticSegmenter._compute_iou(box, bounding_box)
                    for box in reference_bounding_boxes
                ]
            )
            max_overlapping_idx = np.argmax(ious)
            if ious[max_overlapping_idx] >= 0.1:
                instance_number = reference_ids[max_overlapping_idx]
                self._instances_centers[instance_number - 1] = box_center_3d
                if instance_number - 1 <= self._last_ref_center_idx:
                    self._matched_ref_centers[instance_number - 1] = True
                self._current_image_ids.append(instance_number)
                return instance_number

        self._instances_centers.append(box_center_3d)
        instance_number = len(self._instances_centers)
        self._current_image_ids.append(instance_number)
        return instance_number

    def _compute_instance_keypoints_matches(
        self,
        bounding_box,
        kps,
        ref_kps,
        reference_bounding_boxes,
        reference_ids,
        box_center_3d: np.ndarray,
    ):
        # Save the bounding box
        self._current_image_bounding_boxes.append(bounding_box)

        # If this is not the first image
        if len(kps) != 0:
            # Compute keypoints matched with this bounding_box
            matched_keypoints = ref_kps[
                SemanticSegmenter._get_keypoints_in_bounding_box(bounding_box, kps)
            ]

            # Compute the bounding_box idx with more keypoints matched
            matched_bounding_box_idx, n_matches = (
                SemanticSegmenter._get_best_matched_bounding_box(
                    reference_bounding_boxes, matched_keypoints
                )
            )

            # If enough matches, save it
            if n_matches >= 1:
                instance_number = reference_ids[matched_bounding_box_idx]
                self._instances_centers[instance_number - 1] = box_center_3d
                if instance_number - 1 <= self._last_ref_center_idx:
                    self._matched_ref_centers[instance_number - 1] = True
                self._current_image_ids.append(instance_number)
                return instance_number

        self._instances_centers.append(box_center_3d)
        instance_number = len(self._instances_centers)
        self._current_image_ids.append(instance_number)
        return instance_number

    # CONSIDERAZIONE: per ora il metodo che sembra essere piu' promettente
    def _compute_instance_number_centers3d(self, box_center: np.ndarray) -> int:
        if len(self._instances_centers) > 0:
            # First check if we match to a reference center
            reference_centers = self.get_ref_instances_centers()
            if len(reference_centers) > 0:
                distances = np.sum((reference_centers - box_center) ** 2, axis=1)
                min_distance_idx = np.argmin(distances).astype(int)
                if (
                    distances[min_distance_idx]
                    <= self._config.semantic.instance_centers_th + 0.05
                ):
                    self._instances_centers[min_distance_idx] = box_center
                    instance_number = min_distance_idx + 1
                    if instance_number - 1 <= self._last_ref_center_idx:
                        self._matched_ref_centers[instance_number - 1] = True
                    return instance_number

            # Then compute the "nearest" instance between the new centers
            new_centers = self.get_new_instances_centers()
            if len(new_centers) > 0:
                distances = np.sum(
                    (self.get_new_instances_centers() - box_center) ** 2, axis=1
                )
                min_distance_idx = np.argmin(distances).astype(int)
                if (
                    distances[min_distance_idx]
                    <= self._config.semantic.instance_centers_th
                ):
                    self._instances_centers[min_distance_idx] = box_center
                    instance_number = min_distance_idx + 1
                    if instance_number - 1 <= self._last_ref_center_idx:
                        self._matched_ref_centers[instance_number - 1] = True
                    return instance_number

        self._instances_centers.append(box_center)
        return len(self._instances_centers)

    def _save_infos_for_next_frame(self, rgb_img: np.ndarray):
        self._2d_centers.append(
            np.asarray(copy.deepcopy(self._current_image_2d_centers))
        )
        self._bounding_boxes.append(
            np.asarray(copy.deepcopy(self._current_image_bounding_boxes))
        )
        self._image_ids.append(np.asarray(copy.deepcopy(self._current_image_ids)))
        self._current_image_2d_centers = []
        self._current_image_bounding_boxes = []
        self._current_image_ids = []
        self._ref_img = rgb_img

    def _match_images(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._image_matcher.match(img1, img2, visualize=False)

    def _warp_image(
        self, img: np.ndarray, kpts1: np.ndarray, kpts2: np.ndarray
    ) -> np.ndarray:
        M, _ = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 5.0)
        img1_warped = cv2.warpPerspective(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR
        )
        return img1_warped

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray):
        # Calculate intersection
        inter_min = np.maximum(box1[:2], box2[:2])
        inter_max = np.minimum(box1[2:], box2[2:])
        inter_dims = np.maximum(0, inter_max - inter_min)
        inter_area = np.prod(inter_dims)

        # Calculate union
        area1 = np.prod(box1[2:] - box1[:2])
        area2 = np.prod(box2[2:] - box2[:2])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def _get_keypoints_in_bounding_box(bounding_box: np.ndarray, kps: np.ndarray):
        x_min, y_min, x_max, y_max = bounding_box
        return (
            (kps[:, 0] >= x_min)
            & (kps[:, 0] <= x_max)
            & (kps[:, 1] >= y_min)
            & (kps[:, 1] <= y_max)
        )

    @staticmethod
    def _get_best_matched_bounding_box(bounding_boxes, kps) -> Tuple[int, int]:
        best_bbox_idx = -1
        n_best_matches = 0
        for bbox_idx, bbox in enumerate(bounding_boxes):
            inside_bbox2 = SemanticSegmenter._get_keypoints_in_bounding_box(bbox, kps)
            num_points = np.sum(inside_bbox2)
            if num_points > n_best_matches:
                best_bbox_idx = bbox_idx
                n_best_matches = num_points
        return best_bbox_idx, n_best_matches

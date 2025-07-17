# MIT License
#
# Copyright (c) 2024 Luca Lobefaro, Meher V.R. Malladi, Tiziano Guadagnino, Cyrill Stachniss
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
import os
from pathlib import Path
import sys
from typing import Tuple, List, Dict
import numpy as np
import open3d as o3d
import cv2
import json

from st_mapping.tools import (
    load_kitti_poses,
    load_camera_parameters,
)
from st_mapping.core.mapping import PointCloud, point_cloud_from_points_and_colors


class GenericDataset:
    """GenericDataset.
    This dataset expects the following folder structure:
        main_folder:
            > rgb
                0001.jpg
                0002.jpg
                ...
            > depth
                0001.jpg
                0002.jpg
                ...
            poses.txt
            gt_poses.txt
            params.yaml
    If has_poses = False then we do not expect the file poses.txt to be present.
    Note: the file gt_poses.txt is optional, if it is there, then we can evaluate the system
    in visual odometry mode.
    """

    def __init__(
        self,
        main_folder: Path,
        has_poses: bool,
        depth_scale: float,
        degradation_depth_level: int,
        *_,
        **__,
    ):
        self._main_folder = main_folder
        self._has_poses = has_poses
        self._depth_scale = depth_scale

        (
            self._has_gt,
            self._has_map,
            self._has_semantic,
            self._has_boxes,
            self._has_semantic_masks,
            self._has_gt2pred_associations,  # Computed with evaluation intra-sequence
            self._has_gt_map,
        ) = self._check_dataset_consistency()

        self._rgb_filenames, self._depth_filenames, self._n_images = (
            self._load_img_filenames()
        )
        self._intrinsics, self._extrinsics, self._img_width, self._img_heigth = (
            load_camera_parameters(main_folder / "params.yaml")
        )
        self._poses, self._gt_poses = self._load_poses()

        self._degradation_depth_level = degradation_depth_level

    def _check_dataset_consistency(
        self,
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool]:
        subfiles = os.listdir(str(self._main_folder))
        if (
            not "rgb" in subfiles
            or not "depth" in subfiles
            or not os.path.isdir(str(self._main_folder / "rgb"))
            or not os.path.isdir(str(self._main_folder / "depth"))
            or not "params.yaml" in subfiles
        ):
            print(
                "[ERROR] Check that the given dataset path contains rgb/ depth/ folders and params.yaml file"
            )
            sys.exit(1)
        if self._has_poses and not "poses.txt" in subfiles:
            print(
                "[ERROR] No poses in the given dataset. If you are using this for mapping without visual odometry or as a reference map, please compute the poses first."
            )
            sys.exit(1)
        has_semantic = False
        if "labels.txt" in subfiles and "instances_centers.txt" in subfiles:
            has_semantic = True
        has_boxes = False
        if (
            "centers2d.npz" in subfiles
            and "bounding_boxes.npz" in subfiles
            and "image_ids.npz" in subfiles
        ):
            has_boxes = True
        return (
            "gt_poses.txt" in subfiles,
            "map.ply" in subfiles,
            has_semantic,
            has_boxes,
            "instances_masks" in subfiles,
            "gt_mapping.json" in subfiles,
            "gt_map.ply" in subfiles and "gt_labels.txt" in subfiles,
        )

    def _load_img_filenames(self):
        rgb_filenames = os.listdir(str(self._main_folder / "rgb"))
        rgb_filenames.sort()
        depth_filenames = os.listdir(str(self._main_folder / "depth"))
        depth_filenames.sort()
        n_images = len(rgb_filenames)
        if len(depth_filenames) != n_images:
            print("[ERROR] Number of rgb and depth images are not the same.")
            sys.exit(1)
        return rgb_filenames, depth_filenames, n_images

    def _load_poses(self):

        if self._has_poses:
            poses = load_kitti_poses(self._main_folder / "poses.txt")
            if len(poses) != self._n_images:
                print("[ERROR] Number of poses and number of images do not match.")
                sys.exit(1)
        else:
            poses = np.array([])

        if self._has_gt:
            gt_poses = load_kitti_poses(self._main_folder / "gt_poses.txt")
            if len(gt_poses) != self._n_images:
                print(
                    "[ERROR] Number of ground truth poses and number of images do not match."
                )
                sys.exit(1)
        else:
            gt_poses = np.array([])

        return poses, gt_poses

    def __len__(self):
        return self._n_images

    def __getitem__(self, idx):
        return self.get_image(idx)

    def load_map(self) -> PointCloud:
        if self._has_map:
            map_pcd = o3d.io.read_point_cloud(str(self._main_folder / "map.ply"))
        else:
            map_pcd = o3d.geometry.PointCloud()
        return point_cloud_from_points_and_colors(
            np.asarray(map_pcd.points), np.asarray(map_pcd.colors)
        )

    def load_semantic(self) -> Tuple[List[int], List[np.ndarray]]:
        if not self._has_semantic:
            print(
                "[ERORR]: you are trying to load semantic for a dataset that does not have semantics."
            )
            sys.exit(1)
        points_labels = np.loadtxt(self._main_folder / "labels.txt").astype(np.int8)
        instances_centers = np.loadtxt(self._main_folder / "instances_centers.txt")
        instances_centers = [el for el in instances_centers]
        return points_labels.tolist(), instances_centers

    def load_boxes(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if not self._has_boxes:
            print(
                "[ERORR]: you are trying to load 2d centers and boxes for a dataset that does not have them."
            )
            sys.exit(1)
        centers2d = np.load(self._main_folder / "centers2d.npz")
        bounding_boxes = np.load(self._main_folder / "bounding_boxes.npz")
        image_ids = np.load(self._main_folder / "image_ids.npz")
        return (
            [centers2d[key] for key in centers2d],
            [bounding_boxes[key] for key in bounding_boxes],
            [image_ids[key] for key in image_ids],
        )

    def load_gt2pred_associations(self) -> Dict:
        with open(self.get_folder_path() / "gt_mapping.json", "r") as gtmapping_file:
            json_dict = json.load(gtmapping_file)
        gt2pred_associations = {}
        for key, value in json_dict.items():
            gt2pred_associations[int(key)] = value
        return gt2pred_associations

    def get_nearest_image_and_pose(
        self, pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        nearest_idx = 0
        min_distance = float("inf")
        for idx, T in enumerate(self._poses):
            distance = np.linalg.norm(T - pose, "fro")
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
        rgb_img, depth_img = self.get_image(nearest_idx)
        return rgb_img, depth_img, self._poses[nearest_idx], nearest_idx

    def get_folder_path(self) -> Path:
        return self._main_folder

    def get_semantic_mask(self, idx) -> np.ndarray:
        if not self.has_semantic_gt():
            print(
                "[ERROR] you are trying to get instances masks in a dataset that does not contains them"
            )
            exit(1)
        return cv2.imread(
            str(self._main_folder / "instances_masks" / self._rgb_filenames[idx]),
            cv2.IMREAD_UNCHANGED,
        )

    def get_image(
        self,
        idx,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx > self._n_images:
            print(
                f"[ERROR] {idx} is an invalid dataset index, dataset len: {self._n_images}."
            )
        rgb_img = cv2.imread(
            str(self._main_folder / "rgb" / self._rgb_filenames[idx]), cv2.IMREAD_COLOR
        )
        depth_unprocessed = cv2.imread(
            str(self._main_folder / "depth" / self._depth_filenames[idx]),
            cv2.IMREAD_UNCHANGED,
        )
        depth_img = (depth_unprocessed / self._depth_scale).astype(np.float32)
        depth_img = self.degradate_depth_image_(depth_img)
        return rgb_img, depth_img

    def img_width(self):
        return self._img_width

    def img_height(self):
        return self._img_heigth

    def get_pose(self, idx):
        if idx < 0 or idx > len(self._poses):
            print(
                f"[ERROR] {idx} is an invalid pose index (check that this dataset has poses)."
            )
        return self._poses[idx]

    def get_gt_pose(self, idx):
        if idx < 0 or idx > len(self._gt_poses):
            print(
                f"[ERROR] {idx} is an invalid ground truth pose index (check that this dataset has gt_poses)."
            )
        return self._gt_poses[idx]

    def get_gt_map(self) -> Tuple[np.ndarray, np.ndarray]:
        map_gt = o3d.io.read_point_cloud(str(self._main_folder / "gt_map.ply"))
        labels_gt = np.loadtxt(self._main_folder / "gt_labels.txt").astype(np.int8)
        return np.asarray(map_gt.points), labels_gt.tolist()

    def get_poses(self):
        if not self._has_poses:
            print(f"[ERROR] This dataset does not have poses.")
        return self._poses

    def get_gt_poses(self):
        if not self._has_gt:
            print(f"[ERROR] This dataset does not have ground truth poses.")
        return self._gt_poses

    def get_intrinsic(self):
        return self._intrinsics

    def get_extrinsics(self):
        return self._extrinsics

    def has_gt(self):
        return self._has_gt

    def has_semantic_gt(self):
        return self._has_semantic_masks

    def has_gt2pred_associations(self):
        return self._has_gt2pred_associations

    def has_map(self):
        return self._has_map

    def has_gt_map(self):
        return self._has_gt_map

    # This is here for an ablation study
    def degradate_depth_image_(self, depth_img):
        # If noise level is 0, do nothing
        if self._degradation_depth_level == 0:
            return depth_img

        # Initialize noise level to 1
        mean_noise = 0
        std_noise = 0.01  # 1cm
        invalid_prob = 0.05  # 5%

        # Update noise level
        if self._degradation_depth_level == 2:
            std_noise = 0.05  # 5cm
            invalid_prob = 0.1  # 10%
        elif self._degradation_depth_level == 3:
            std_noise = 0.1  # 10cm
            invalid_prob = 0.3  # 30%

        # Apply Gaussian noise
        noise = np.random.normal(mean_noise, std_noise, depth_img.shape)
        noisy_depth = depth_img.astype(np.float32) + noise
        noisy_depth = np.clip(noisy_depth, 0, np.max(depth_img)).astype(depth_img.dtype)

        # Clip random pixels
        mask = np.random.rand(*depth_img.shape) < invalid_prob
        noisy_depth[mask] = np.nan

        return noisy_depth

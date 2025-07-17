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
from st_mapping.config.config import StMappingConfig
from st_mapping.tools import (
    save_kitti_poses,
    save_point_cloud,
    PrettyResults,
    run_semantic_evaluation_intersequences,
    evaluate_map_associations,
    pred2gt_associations,
)
from st_mapping.odometry import OdometryOnRef
from st_mapping.semantic_segmenter import SemanticSegmenter
from st_mapping.core.metrics import sequence_error, absolute_trajectory_error
from st_mapping.core.mapping import extract_point_cloud
from st_mapping.core.global_map import GlobalMap
from tqdm.auto import trange
import time

from st_mapping.tools.visualizer import (
    StubVisualizer,
    SemanticMappingVisualizer,
)


class SemanticMappingOnRefPipeline:

    # --- PUBLIC INTERFACE ---------------------------------------------------------------
    def __init__(self, dataset, ref_dataset, config: StMappingConfig, visualize: bool):
        self._dataset = dataset
        self._ref_dataset = ref_dataset
        self._config = config

        self._global_map = GlobalMap()
        self._poses = []
        self._exec_times = []
        self._results = PrettyResults()
        self._semantic_masks = []

        self._n_frames = len(dataset)
        if self._config.n_frames > 1 and self._config.n_frames < self._n_frames:
            self._n_frames = self._config.n_frames

        # Load ref map
        self._ref_pcd = self._ref_dataset.load_map()
        self._ref_points_labels, self._ref_instances_centers = (
            self._ref_dataset.load_semantic()
        )

        # Prepare tools
        self._odometry = OdometryOnRef(
            config,
            self._ref_pcd,
            self._ref_dataset.get_pose(0),
            self._dataset.get_extrinsics(),
        )
        self._semantic_segmenter = SemanticSegmenter(
            self._dataset,
            self._config,
            self._odometry,
            self._ref_dataset,
            self._ref_instances_centers,
        )
        self._visualizer = (
            SemanticMappingVisualizer() if visualize else StubVisualizer()
        )
        self._visualizer.register_refmap(
            self._ref_pcd, self._ref_points_labels, self._ref_instances_centers
        )

    def run(self):
        self._run_pipeline()
        self._save_results()
        self._run_evaluation()
        self._results.print()
        self._visualizer.keep_running()
        return

    # --- PRIVATE INTERFACE ---------------------------------------------------------------
    def _run_pipeline(self):
        for idx in trange(0, self._n_frames, unit="frames", dynamic_ncols=True):
            rgb_img, depth_img = self._dataset[idx]
            start_time = time.perf_counter_ns()

            labeled_rgb_img, semantic_mask = self._semantic_segmenter.extract_semantics(
                rgb_img, depth_img
            )

            frame, points_labels = extract_point_cloud(
                rgb_img,
                depth_img,
                semantic_mask,
                self._dataset.get_intrinsic(),
                self._dataset.get_extrinsics(),
                self._config.dataset.depth_min_th,
                self._config.dataset.depth_max_th,
                self._config.dataset.image_stride,
            )
            processed_frame, processed_points_labels, pose = (
                self._odometry.register_frame(frame, points_labels)
            )
            self._global_map.integrate_point_cloud(
                processed_frame, processed_points_labels, pose
            )
            self._exec_times.append(time.perf_counter_ns() - start_time)
            self._poses.append(pose)
            self._visualizer.update(
                pose @ self._dataset.get_extrinsics(),
                labeled_rgb_img,
                self._global_map,
                self._semantic_segmenter,
            )
            self._update_evalutation_results(semantic_mask)

    def _update_evalutation_results(self, semantic_mask: np.ndarray):
        if self._dataset.has_semantic_gt():
            self._semantic_masks.append(semantic_mask)

    def _save_results(self):
        poses_path = self._dataset.get_folder_path() / "poses.txt"
        save_kitti_poses(poses_path, np.array(self._poses))
        print("Poses saved at:", poses_path)
        pcd_path = self._dataset.get_folder_path() / "map.ply"
        points, colors, labels = self._global_map.get_points_colors_and_labels()
        save_point_cloud(pcd_path, points, colors)
        print("Map saved at", pcd_path)
        labels_path = self._dataset.get_folder_path() / "labels.txt"
        np.savetxt(labels_path, np.asarray(labels))
        print("Labels saved at", labels_path)
        centers_path = self._dataset.get_folder_path() / "instance_centers.txt"
        np.savetxt(centers_path, self._semantic_segmenter.get_all_instances_centers())
        print("Instances centers saved at", centers_path)

    def _run_evaluation(self):
        # Run estimation metrics evaluation, only when GT data was provided
        if self._dataset.has_gt():
            avg_tra, avg_rot = sequence_error(self._dataset.get_gt_poses(), self._poses)
            ate_rot, ate_tra = absolute_trajectory_error(
                self._dataset.get_gt_poses(), self._poses
            )
            self._results.append(
                desc="Relative Translation Error(RPE)", units="%", value=avg_tra
            )
            self._results.append(
                desc="Relative Rotational Error(RRE)", units="deg/100m", value=avg_rot
            )
            self._results.append(
                desc="Absolute Trajectory Error (ATE)", units="m", value=ate_tra
            )
            self._results.append(
                desc="Absolute Rotational Error (ARE)", units="rad", value=ate_rot
            )

        # Run timing metrics evaluation, always
        def _get_fps():
            total_time_s = sum(self._exec_times) * 1e-9
            return float(len(self._exec_times) / total_time_s)

        avg_fps = int(np.ceil(_get_fps()))
        avg_ms = int(np.ceil(1e3 * (1 / _get_fps())))
        self._results.append(
            desc="Average Frequency", units="Hz", value=avg_fps, trunc=True
        )
        self._results.append(
            desc="Average Runtime", units="ms", value=avg_ms, trunc=True
        )

        if (
            self._dataset.has_semantic_gt()
            and self._ref_dataset.has_gt2pred_associations()
        ):
            semantic_metrics = run_semantic_evaluation_intersequences(
                self._dataset,
                self._semantic_masks,
                self._ref_dataset.load_gt2pred_associations(),
            )
            self._results.append(
                desc="detection_recall (TAB IV)",
                units="%",
                value=semantic_metrics["detection_recall"] * 100,
                trunc=False,
            )
            self._results.append(
                desc="detection_accuracy  (TAB IV)",
                units="%",
                value=semantic_metrics["detection_accuracy"] * 100,
                trunc=False,
            )
            # self._results.append(
            #     desc="detection_precision",
            #     units="%",
            #     value=semantic_metrics["detection_precision"] * 100,
            #     trunc=False,
            # )
            self._results.append(
                desc="detection_f1_score (TAB IV)",
                units="%",
                value=semantic_metrics["detection_f1_score"] * 100,
                trunc=False,
            )
            self._results.append(
                desc="associations_recall (TAB IV)",
                units="%",
                value=semantic_metrics["associations_recall"] * 100,
                trunc=False,
            )
            self._results.append(
                desc="associations_f1_score (TAB IV)",
                units="%",
                value=semantic_metrics["associations_f1_score"] * 100,
                trunc=False,
            )
            self._results.append(
                desc="tp_association (TAB IV)",
                units="#",
                value=semantic_metrics["tp_association"],
                trunc=False,
            )
            self._results.append(
                desc="fn_association (TAB IV)",
                units="#",
                value=semantic_metrics["fn_association"],
                trunc=False,
            )

        if self._dataset.has_gt_map() and self._ref_dataset.has_gt_map():
            # Get elements useful for evaluation
            points, _, labels = self._global_map.get_points_colors_and_labels()
            ref_points, _ = self._ref_dataset.load_map().get_points_and_colors()
            ref_labels, _ = self._ref_dataset.load_semantic()

            # Load the ground truths
            points_gt, labels_gt = self._dataset.get_gt_map()
            ref_points_gt, ref_labels_gt = self._ref_dataset.get_gt_map()

            # Evaluate associations against baselines (IoU 25%)
            pred2gt = pred2gt_associations(points, points_gt, labels, labels_gt, 0.25)
            ref_pred2gt = pred2gt_associations(
                ref_points, ref_points_gt, ref_labels, ref_labels_gt, 0.25
            )
            matching_results = evaluate_map_associations(pred2gt, ref_pred2gt)
            self._results.append(
                desc="F1 (TAB I)",
                units="%",
                value=matching_results["f1"] * 100,
                trunc=False,
            )
            self._results.append(
                desc="ACCURACY (TAB I)",
                units="%",
                value=matching_results["acc"] * 100,
                trunc=False,
            )

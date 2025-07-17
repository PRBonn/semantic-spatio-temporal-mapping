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
import copy
from abc import ABC
import polyscope as ps
import polyscope.imgui as gui
import cv2

import numpy as np
from st_mapping.tools.visualization_tools import generate_colors_map

# Buttons names
START_BUTTON = "START [SPACE]"
PAUSE_BUTTON = "PAUSE [SPACE]"
NEXT_FRAME_BUTTON = "NEXT FRAME [N]"
QUIT_BUTTON = "QUIT [Q]"

# Colors
BACKGROUND_COLOR = [0.8470, 0.8588, 0.8863]
BACKGROUND_COLOR = [1.0, 1.0, 1.0]
CAMERA_COLOR = [1.0, 0.0, 0.0]

# Size constants
CLOUD_POINT_SIZE = 0.005
POINTS_SIZE_STEP = 0.001
POINTS_SIZE_MIN = 0.001
POINTS_SIZE_MAX = 0.01
INSTANCES_CENTERS_SIZE = 0.01
LINES_SIZE = 0.003

# Other constants
REF_TRANSLATION = 1.2


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, pose, rgb_img, local_map_pcd, semantic_segmenter):
        pass

    def register_refmap(self, ref_pcd, points_labels, ref_instance_centers):
        pass

    def keep_running(self):
        pass

    def quit(self):
        pass


class SemanticMappingVisualizer(StubVisualizer):
    # --- PUBLIC INTERFACE ---------------------------------------------------------------
    def __init__(self):

        # Initilize attributes
        self._camera_poses = []
        self._ref_instances_centers = []
        self._n_ref_instances = 0

        # Initilize GUI controls and attributes
        self._play_mode: bool = False
        self._block_execution: bool = True
        self._points_size: float = CLOUD_POINT_SIZE
        self._points_transparency: float = 1.0
        self._colors_map = generate_colors_map(400)

        # Initialize visualizer
        self._initialize_visualizer()

    def update(self, pose, rgb_img, local_map_pcd, semantic_segmenter):
        # Visualize the image
        self._visualize_image(rgb_img)

        # Visualize camera pose
        self._update_camera_pose(pose)

        # Visualize local map point cloud
        self._visualize_point_cloud(local_map_pcd)

        # Visualize instances centers
        self._visualize_new_instances_centers(
            semantic_segmenter.get_new_instances_centers(),
        )
        self._visualize_matched_instances_centers(
            semantic_segmenter.get_matched_instances_indices()
        )

        # Visualization loop
        self._update_visualizer()

    def register_refmap(self, ref_pcd, points_labels, ref_instance_centers):
        # Initialization
        points, colors = ref_pcd.get_points_and_colors()
        self._ref_instances_centers = np.asarray(copy.deepcopy(ref_instance_centers))
        self._n_ref_instances = len(ref_instance_centers)

        # Translate for visualization
        points[:, 2] += REF_TRANSLATION
        self._ref_instances_centers[:, 2] += REF_TRANSLATION

        # Register cloud
        instances_colors = np.asarray(
            [
                self._colors_map.get(label, colors[idx])
                for idx, label in enumerate(points_labels)
            ]
        )
        cloud = ps.register_point_cloud(
            "ref_map",
            points,
            point_render_mode="quad",
            transparency=self._points_transparency,
        )
        cloud.add_color_quantity("colors", instances_colors, enabled=True)
        cloud.set_radius(self._points_size, relative=False)

        # Register centers
        centers_colors = np.asarray(
            [
                self._colors_map.get(idx + 1)
                for idx in range(0, len(self._ref_instances_centers))
            ]
        )
        instances_centers_cloud = ps.register_point_cloud(
            "ref_instances_centers", self._ref_instances_centers
        )
        instances_centers_cloud.add_color_quantity(
            "colors", centers_colors, enabled=True
        )
        instances_centers_cloud.set_radius(INSTANCES_CENTERS_SIZE, relative=False)

    def keep_running(self):
        cv2.destroyAllWindows()
        self._play_mode = False
        self._block_execution = True
        ps.get_camera_view("camera").set_enabled(False)
        ps.set_user_callback(self._keep_running_callback)
        ps.show()

    # --- PRIVATE INTERFACE ---------------------------------------------------------------
    def _initialize_visualizer(self):
        ps.set_program_name("Semantic Spatio-Temporal Mapping Visualizer")
        ps.init()
        ps.set_background_color(BACKGROUND_COLOR)
        ps.set_verbosity(0)
        ps.set_ground_plane_mode("none")
        ps.set_user_callback(self._main_gui_callback)
        ps.set_build_default_gui_panels(False)
        cv2.namedWindow("Image Stream")

    def _visualize_image(self, img):
        # Visualize image
        rgb_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Image Stream", rgb_img)
        cv2.setWindowProperty("Image Stream", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)

    def _update_camera_pose(self, pose):
        self._camera_poses.append(pose)
        camera_params = ps.CameraParameters(
            ps.CameraIntrinsics(fov_vertical_deg=30, aspect=2),
            ps.CameraExtrinsics(mat=np.linalg.inv(pose)),
        )
        new_cam = ps.register_camera_view("camera", camera_params)
        new_cam.set_widget_color(CAMERA_COLOR)

    def _visualize_point_cloud(self, local_map_pcd):
        points, colors, labels = local_map_pcd.get_points_colors_and_labels()

        # Color the instances
        instances_colors = np.asarray(
            [
                self._colors_map.get(label, colors[idx])
                for idx, label in enumerate(labels)
            ]
        )

        # Register the cloud
        cloud = ps.register_point_cloud(
            "local_map",
            points,
            point_render_mode="quad",
            transparency=self._points_transparency,
        )
        cloud.add_color_quantity("colors", instances_colors, enabled=True)
        cloud.set_radius(self._points_size, relative=False)

    def _visualize_new_instances_centers(
        self,
        instances_centers,
    ):
        if instances_centers.shape[0] == 0:
            return

        # Generate colors
        centers_colors = np.asarray(
            [
                self._colors_map.get(idx + 1)
                for idx in range(
                    self._n_ref_instances,
                    len(instances_centers) + self._n_ref_instances,
                )
            ]
        )

        # Visualize centers
        instances_centers_cloud = ps.register_point_cloud(
            "new_instances_centers", np.asarray(instances_centers)
        )
        instances_centers_cloud.add_color_quantity(
            "colors", centers_colors, enabled=True
        )
        instances_centers_cloud.set_radius(INSTANCES_CENTERS_SIZE, relative=False)

    def _visualize_matched_instances_centers(self, matched_indices):
        if matched_indices.shape[0] == 0:
            return

        matched_centers = self._ref_instances_centers[matched_indices]
        matched_centers[:, 2] -= REF_TRANSLATION

        # Generate colors
        centers_colors = np.asarray(
            [self._colors_map.get(idx + 1) for idx in matched_indices]
        )

        # Visualize matched centers
        instances_centers_cloud = ps.register_point_cloud(
            "matched_instances_centers", matched_centers
        )
        instances_centers_cloud.add_color_quantity(
            "colors", centers_colors, enabled=True
        )
        instances_centers_cloud.set_radius(INSTANCES_CENTERS_SIZE, relative=False)

        # Visualize connections
        nodes = np.concatenate(
            (matched_centers, self._ref_instances_centers[matched_indices])
        )
        n_matches = matched_centers.shape[0]
        edges = np.asarray([(idx, idx + n_matches) for idx in range(n_matches)])
        matches_curve = ps.register_curve_network("matches", nodes, edges)
        matches_curve.add_color_quantity(
            "matches_color", centers_colors, defined_on="edges", enabled=True
        )
        matches_curve.set_radius(LINES_SIZE, relative=False)

    def _update_visualizer(self):
        while self._block_execution:
            ps.frame_tick()
            if self._play_mode:
                break
        self._block_execution = not self._block_execution

    # --- GUI Callbacks ------------------------------------------------------------------
    def _start_pause_callback(self):
        button_name = PAUSE_BUTTON if self._play_mode else START_BUTTON
        if gui.Button(button_name) or gui.IsKeyPressed(gui.ImGuiKey_Space):
            self._play_mode = not self._play_mode

    def _next_frame_callback(self):
        if gui.Button(NEXT_FRAME_BUTTON) or gui.IsKeyPressed(gui.ImGuiKey_N):
            self._block_execution = not self._block_execution

    def _points_size_callback(self):
        key_changed = False
        if gui.IsKeyPressed(gui.ImGuiKey_Minus):
            self._points_size = max(
                POINTS_SIZE_MIN, self._points_size - POINTS_SIZE_STEP
            )
            key_changed = True
        if gui.IsKeyPressed(gui.ImGuiKey_Equal):
            self._points_size = min(
                POINTS_SIZE_MAX, self._points_size + POINTS_SIZE_STEP
            )
            key_changed = True
        changed, self._points_size = gui.SliderFloat(
            "Points Size",
            self._points_size,
            v_min=POINTS_SIZE_MIN,
            v_max=POINTS_SIZE_MAX,
        )
        if changed or key_changed:
            ps.get_point_cloud("local_map").set_radius(
                self._points_size, relative=False
            )
            if ps.has_point_cloud("ref_map"):
                ps.get_point_cloud("ref_map").set_radius(
                    self._points_size, relative=False
                )

    def _points_transparency_callback(self):
        changed, self._points_transparency = gui.SliderFloat(
            "Points Transparency",
            self._points_transparency,
            v_min=0,
            v_max=1.0,
        )
        if changed:
            ps.get_point_cloud("local_map").set_transparency(self._points_transparency)
            if ps.has_point_cloud("ref_map"):
                ps.get_point_cloud("ref_map").set_transparency(
                    self._points_transparency
                )

    def _quit_callback(self):
        posX = (
            gui.GetCursorPosX()
            + gui.GetColumnWidth()
            - gui.CalcTextSize(QUIT_BUTTON)[0]
            - gui.GetScrollX()
            - gui.ImGuiStyleVar_ItemSpacing
        )
        gui.SetCursorPosX(posX)
        if (
            gui.Button(QUIT_BUTTON)
            or gui.IsKeyPressed(gui.ImGuiKey_Escape)
            or gui.IsKeyPressed(gui.ImGuiKey_Q)
        ):
            print("Visualizer Bye Bye!")
            ps.unshow()
            cv2.destroyAllWindows()
            os._exit(0)

    def _scene_options_callback(self):
        gui.TextUnformatted("Scene Options:")
        self._points_size_callback()
        self._points_transparency_callback()

    def _main_gui_callback(self):
        self._start_pause_callback()
        if not self._play_mode:
            gui.SameLine()
            self._next_frame_callback()
        gui.Separator()
        self._scene_options_callback()
        gui.Separator()
        self._quit_callback()

    def _keep_running_callback(self):
        self._scene_options_callback()
        gui.Separator()
        self._quit_callback()

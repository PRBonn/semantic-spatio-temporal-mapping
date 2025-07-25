// MIT License
//
// Copyright (c) 2024 Luca Lobefaro, Meher V.R. Malladi, Tiziano Guadagnino, Cyrill Stachniss
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Dense>
#include <forward_list>
#include <sophus/se3.hpp>

#include "Mapping.hpp"

namespace st_mapping {
class GlobalMap {
public:
    GlobalMap() : _n_points(0){};
    GlobalMap(const GlobalMap &other) = delete;
    GlobalMap(GlobalMap &&other) = default;
    GlobalMap &operator=(const GlobalMap &other) = delete;
    GlobalMap &operator=(GlobalMap &&other) = default;

    void IntegratePointCloud(const PointCloud &pcd,
                             const std::vector<int> &points_labels,
                             const Sophus::SE3d &T);

    PointsAndColor GetPointsAndColors() const;
    PointsColorsAndLabels GetPointsColorsAndLabels() const;

private:
    std::forward_list<PointWithColor> _points;
    std::forward_list<int> _labels;
    size_t _n_points;
};
}  // namespace st_mapping

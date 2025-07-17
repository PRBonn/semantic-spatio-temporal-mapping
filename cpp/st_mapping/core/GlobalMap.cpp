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
#include "GlobalMap.hpp"

#include <algorithm>

#include "Mapping.hpp"

namespace st_mapping {

void GlobalMap::IntegratePointCloud(const PointCloud &pcd,
                                    const std::vector<int> &points_labels,
                                    const Sophus::SE3d &T) {
    // Initialization
    std::vector<int> indices(pcd.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Save cloud
    std::for_each(indices.cbegin(), indices.cend(), [&](const int idx) {
        const auto &[point, color] = pcd[idx];
        Eigen::Vector3d transformed_point = T * point;
        _points.push_front({transformed_point, color});
        _labels.push_front(points_labels[idx]);
        _n_points++;
    });
}

PointsAndColor GlobalMap::GetPointsAndColors() const {
    std::vector<Eigen::Vector3d> points, colors;
    points.reserve(_n_points);
    colors.reserve(_n_points);
    std::for_each(_points.cbegin(), _points.cend(), [&](const PointWithColor &colored_point) {
        const auto &[point, color] = colored_point;
        points.emplace_back(point);
        colors.emplace_back(color);
    });
    return std::make_pair(points, colors);
}

PointsColorsAndLabels GlobalMap::GetPointsColorsAndLabels() const {
    std::vector<Eigen::Vector3d> points, colors;
    std::vector<int> labels{_labels.cbegin(), _labels.cend()};
    points.reserve(_n_points);
    colors.reserve(_n_points);
    std::for_each(_points.cbegin(), _points.cend(), [&](const PointWithColor &colored_point) {
        const auto &[point, color] = colored_point;
        points.emplace_back(point);
        colors.emplace_back(color);
    });
    return std::make_tuple(points, colors, labels);
}

}  // namespace st_mapping

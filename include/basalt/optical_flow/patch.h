/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <Eigen/Dense>

#include <basalt/image/image.h>
#include <basalt/optical_flow/patterns.h>

namespace basalt {

template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;

  typedef Eigen::Matrix<int, 2, 1> Vector2i;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 1, 2> Vector2T;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 2> MatrixP2;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<Scalar, 3, PATTERN_SIZE> Matrix3P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 4> MatrixP4;
  typedef Eigen::Matrix<int, 2, PATTERN_SIZE> Matrix2Pi;

  static const Matrix2P pattern2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() { mean = 0; }

  OpticalFlowPatch(const Image<const uint16_t> &img, const Vector2 &pos) {
    setFromImage(img, pos);
  }

  void setFromImage(const Image<const uint16_t> &img, const Vector2 &pos) {
    this->pos = pos;

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector2 grad_sum(0, 0);

    MatrixP2 grad; // hm: matrix of patter_size * 2, stores the gradient, with respect to x (right) and y (down), for data!

    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p = pos + pattern2.col(i);
      if (img.InBounds(p, 2)) {
        // hm: interpolated gradient value, first element of the vector is the interpolation, the next two is dx, dy
        Vector3 valGrad = img.interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        sum += valGrad[0];
        grad.row(i) = valGrad.template tail<2>(); // hm: stores gradient of pixels in each row
        grad_sum += valGrad.template tail<2>(); // hm: to calculate average gradient, sum them up here first
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;

    Scalar mean_inv = num_valid_points / sum;
    
    // hm: NOTE that the SE2 here applies to the template (data), not the wrapped image
    Eigen::Matrix<Scalar, 2, 3> Jw_se2; // hm: each column, x, y, theta. x points right, y points down
    Jw_se2.template topLeftCorner<2, 2>().setIdentity();

    MatrixP3 J_se2; // hm: pattern_size * 3, account for 2 translation and 1 rotation. reflects how each residual component changes, with regards to SE(2)

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) {
        const Scalar data_i = data[i];
        const Vector2 grad_i = grad.row(i);
        // hm: grad_i is the current pixel's gradient
        // hm: effectively = grad_i / averaged_intensity -  averaged_grad * grad_i / averaged_intensity
        grad.row(i) =
            num_valid_points * (grad_i * sum - grad_sum * data_i) / (sum * sum);

        data[i] *= mean_inv; // hm: divide by mean for all pixels
      } else {
        grad.row(i).setZero();
      }

      // Fill jacobians with respect to SE2 warp
      // hm: the image is store in a row-major fashion, starting from the top-left corner. x correspon
      Jw_se2(0, 2) = -pattern2(1, i); // hm: change of x, respect to rotation. Apply small angle differentiation, give dx/dtheta = -y
      Jw_se2(1, 2) = pattern2(0, i); // hm: change of y, respect to rotation. Similarly dy/dtheta = x
      // hm: J_se2 = di/dw * dw/dse2 = grad * Jw_se2
      J_se2.row(i) = grad.row(i) * Jw_se2; // hm: jecobian of each pixel = 
    }

    // hm: formulation refer to Usenko(2019), Preliminaries, with W = identity
    Matrix3 H_se2 = J_se2.transpose() * J_se2;
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv); // hm: this is to solve the inverse. This is because H_se2_inv is set to identity. so the result x IS THE INVERSE

    H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose();
  }

  // hm: transformed_pattern is rotated and shifted outside this call
  // hm: residual is computed against the data vector, size patter_size
  inline bool residual(const Image<const uint16_t> &img,
                       const Matrix2P &transformed_pattern,
                       VectorP &residual) const {
    Scalar sum = 0; // hm: stores the sum of image pixels, using valid points in the transformed pattern
    Vector2 grad_sum(0, 0);
    int num_valid_points = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (img.InBounds(transformed_pattern.col(i), 2)) {
        residual[i] = img.interp<Scalar>(transformed_pattern.col(i));
        sum += residual[i];
        num_valid_points++;
      } else {
        residual[i] = -1;
      }
    }

    int num_residuals = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        // hm: sum / num_valid_points gives the averaged intensity of the patch
        residual[i] = num_valid_points * val / sum - data[i];
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }

    // hm: num_residuals indicates the number of points that both valid in data and residuals
    return num_residuals > PATTERN_SIZE / 2;
  }

  Vector2 pos;
  VectorP data;  // negative if the point is not valid

  // MatrixP3 J_se2;  // total jacobian with respect to se2 warp
  // Matrix3 H_se2_inv;
  Matrix3P H_se2_inv_J_se2_T;

  Scalar mean; // hm: mean of data, before deviding by mean
};

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P
    OpticalFlowPatch<Scalar, Pattern>::pattern2 = Pattern::pattern2;

}  // namespace basalt

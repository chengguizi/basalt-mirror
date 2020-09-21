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

#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

#include <tbb/parallel_for.h>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <sstream>
namespace basalt {

template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  FrameToFrameOpticalFlow(const VioConfig& config,
                          const basalt::Calibration<double>& calib)
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), config(config) {
    input_queue.set_capacity(100);

    this->calib = calib.cast<Scalar>();

    patch_coord = PatchT::pattern2.template cast<float>();

    if (calib.intrinsics.size() > 1) {
      Eigen::Matrix4d Ed;
      Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, Ed);
      std::cout<<"cam1 to cam0 : " << T_i_j.translation().x() << ","
              << T_i_j.translation().y() << ","
              << T_i_j.translation().z() << ","
              << T_i_j.unit_quaternion().x() << ","
              << T_i_j.unit_quaternion().y() << ","
              << T_i_j.unit_quaternion().z() << ","
              << T_i_j.unit_quaternion().w() << std::endl<<std::endl;
      E = Ed.cast<Scalar>();
    }

    processing_thread.reset(
        new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));
  }

  ~FrameToFrameOpticalFlow() { processing_thread->join(); }

  void processingLoop() {
    OpticalFlowInput::Ptr input_ptr;

    while (true) {
      input_queue.pop(input_ptr);

      // hm: add logic for realtime requirement
      if (config.vio_enforce_realtime){
        int skipped_image = 0;
        while (input_queue.try_pop(input_ptr)) {skipped_image++;}
        if(skipped_image)
          std::cerr<< "[Optical Flow Warning] skipped input image size: "<<skipped_image<<std::endl;
      }

      if (!input_ptr.get()) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }

      processFrame(input_ptr->t_ns, input_ptr);
    }
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    // hm: first frame
    if (t_ns < 0) {
      // std::cout<<"first t_ns: "<<t_ns<<std::endl<<std::endl;
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      // hm: size depends on the number of cameras
      transforms->observations.resize(calib.intrinsics.size());
      transforms->t_ns = t_ns;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<u_int16_t>>);
      pyramid->resize(calib.intrinsics.size());

      // hm: Running pyramid for all images in parallel
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      transforms->input_images = new_img_vec;

      // hm: detect keypoints in the first image, at level 0
      addPoints();
      filterPoints();

    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<u_int16_t>>);
      pyramid->resize(calib.intrinsics.size());
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;

      // hm: perform feature tracking for each camera separately
      //Yu: track from previous frame to current ones
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        if(config.vio_debug){
          std::cout<<"cam"<<i<<" track from previous frame to current one"<<std::endl;

        }
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);

        if(config.vio_debug){
          std::cout<<"cam"<<i<<"_pre_points: "<< transforms->observations.at(i).size()<<", tracked points from previous frame: "<<new_transforms->observations[i].size()<<" track ratio: "<< float(new_transforms->observations[i].size())/transforms->observations.at(i).size()<<std::endl;
        }
      }

      transforms = new_transforms;
      transforms->input_images = new_img_vec;
      //Yu: detect new points in cam0, and track from cam0 to cam1
      //Yu: check out why this step has the most reject points
      addPoints();
      // hm: here uses CAMERA MODEL!
      // Yu: because addpoints uses optical flow to track
      // so here we need to check if the tracked points valid the epipolar constraint
      filterPoints();

      //draw matching points
      if(config.feature_match_show){
        std::vector<cv::KeyPoint> kp1, kp2, kp0;
        std::vector<cv::DMatch> match;
        int match_id = 0;
        basalt::Image<const uint16_t> img_raw_1(pyramid->at(0).lvl(1)), img_raw_2(pyramid->at(1).lvl(1));
        int w = img_raw_1.w; 
        int h = img_raw_1.h;
        cv::Mat img1(h, w, CV_8U);
        cv::Mat img2(h, w, CV_8U);
        for(int y = 0; y < h; y++){
          uchar* sub_ptr_1 = img1.ptr(y);
          uchar* sub_ptr_2 = img2.ptr(y);

          for(int x = 0; x < w; x++){
            sub_ptr_1[x] = (img_raw_1(x,y) >> 8);
            sub_ptr_2[x] = (img_raw_2(x,y) >> 8);

          }
        }

        for(const auto& kv: transforms->observations[0]){
          auto it = transforms->observations[1].find(kv.first);
          if(it != transforms->observations[1].end()){
            
            kp1.push_back(cv::KeyPoint(cv::Point2f(kv.second.translation()[0]/2, kv.second.translation()[1]/2), 1));
            kp2.push_back(cv::KeyPoint(cv::Point2f(it->second.translation()[0]/2, it->second.translation()[1]/2), 1));
            match.push_back(cv::DMatch(match_id,match_id,1));
            match_id++;
          }
          else{
            kp0.push_back(cv::KeyPoint(cv::Point2f(kv.second.translation()[0]/2, kv.second.translation()[1]/2), 1));
          }
        }
        cv::Mat image_show(h, w*2, CV_8U);
        cv::drawKeypoints(img1, kp0,img1);
        cv::drawMatches(img1,kp1,img2,kp2,match, image_show);
        cv::imshow("matching result", image_show);
        cv::waitKey(1);
      }

    }

    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      if(!output_queue->try_push(transforms)){
          std::cout<<"frame to frame optical flow output queue is full: "<<output_queue->size()<<std::endl;
          abort();
        }
    }

    frame_counter++;
  }

  void trackPoints(const basalt::ManagedImagePyr<u_int16_t>& pyr_1,
                   const basalt::ManagedImagePyr<u_int16_t>& pyr_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_2, bool leftToRight = false) const {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto& kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f> result;
    int cntValidTrack{0}, cntValidLnR{0};
    
    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      /*yu: check which part is easy to fail in left to right tracking
      Three steps to check:
      1. trackPointAtLevel: Check one pyr level: if the residul = patch - data, is valid
          valid means that 1.1 at least PATTERN_SIZE / 2 number of points that both valid in template and residuals patch
                           1.2 the points transformed by the updated and optimized T can still be seen in the image 
      2. trackPoint: check if the traking is valid at all pyr levels
      3. This Func: square norm diff from "left to right transform" and "right to left transform" should be less than config.optical_flow_max_recovered_dist2
      */
      
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;
        // if(leftToRight)
        //   transform_2 = *transform_1

        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);


        if (valid) {
          Eigen::AffineCompact2f transform_1_recovered = transform_2;

          // hm: validate the point could be tracked in reverse, from current to previous
          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);

          if (valid) {
            cntValidTrack++;
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();
            // if(config.vio_debug){
            //   std::stringstream ss;
            //   if(dist2>500){
            //     ss<<"dist2: "<<dist2<<std::endl;
            //     ss<<"transform_1: "<<transform_1.translation()<<std::endl;
            //     ss<<"transform_1_recovered: "<<transform_1_recovered.translation()<<std::endl;
            //   }
            //   std::cout<<ss.str()<<std::endl;
            // }
            if (dist2 < config.optical_flow_max_recovered_dist2) {
              cntValidLnR++;
              result[id] = transform_2;
            }
          }
        }
      }

    };

    tbb::blocked_range<size_t> range(0, num_points);
    
    tbb::parallel_for(range, compute_func);
    // compute_func(range);
    //Yu: try to track the low left to right track ratio reason
    if(config.vio_debug){
        std::cout<<num_points<<" total features from cam0 to track. "
        <<" step1 valid: "<< "to do"
        <<" step2 valid: "<< float(cntValidTrack)/num_points
        <<" step3 valid: "<< float(cntValidLnR)/cntValidTrack<<std::endl;
        
      }

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }
// Yu: track all layers (subfunction)
  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t>& old_pyr,
                         const basalt::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid;
         level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;
      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      // Perform tracking on current level
      patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

// hm: this function only check the points are in the rectangle of the image, not necessarily within the lens circular area
  inline bool trackPointAtLevel(const Image<const u_int16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < config.optical_flow_max_iterations;
         iteration++) {
      typename PatchT::VectorP res; // hm: size of patch_size * 1, column vector

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2; // hm:: pattern2 is a matrix with two rows, xs and ys
      transformed_pat.colwise() += transform.translation(); // hm: R * (x;y) + t

      // hm: res = patch - data
      // Yu: res = patch/average - data(which is also intensity/average for each pixel)
      bool valid = dp.residual(img_2, transformed_pat, res);
      /// Yu: here valid means half of the pixel are within bound (the bound is wxh rectangular) in both image and template patchs 
      if (valid) {
        // hm: SE2 representation, 2 elements for translation, 1 for rotation
        // hm: http://fourier.eng.hmc.edu/e176/lectures/NM/node36.html

        // hm: PROBLEM: sometimes the 3rd element, rotation is not a number        
        Vector3 inc = dp.H_se2_inv_J_se2_T * res; // hm: H_se2_inv_J_se2_T conprises of the update needed to make I(x) of data bigger, which in turn makes residual smaller as r = y - f(x) = img - data
        
        if(res.hasNaN()){
          std::cout << "THERE IS NAN IN THE RESULT" << std::endl;

          std::cout << "transform :" << std::endl << transform.matrix() << std::endl;

          std::cout << "res: " << res.transpose() << std::endl;

          std::abort();
        }

        transform *= SE2::exp(-inc).matrix(); 

        // const int filter_margin = 2;

        if (!calib.intrinsics[0].inBound(transform.translation()))
        // if (!img_2.InBounds(transform.translation(), filter_margin))
          patch_valid = false;
      } else {
        patch_valid = false;
      }
    }

    return patch_valid;
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    // hm: add previously tracked points from the first image to pts0 variable
    // hm: trackPoints() must be executed before this function call
    for (const auto& kv : transforms->observations.at(0)) {
      pts0.emplace_back(kv.second.translation().cast<double>());
    }

    KeypointsData kd;
    
    // hm: pts0 is the current points
    detectKeypoints(pyramid->at(0).lvl(0), kd,
                    config.optical_flow_detection_grid_size, 1, pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;


    //Yu: if the keypoint is out of valid boundry, then we need to drop it directly here.
    //*********************************************************
      std::set<int> kp_to_remove;
      for (size_t i = 0; i < kd.corners.size(); i++) {
        if (!calib.intrinsics[0].inBound(kd.corners[i].cast<Scalar>()))
          kp_to_remove.emplace(i);
      }
      //*************************************************************

    if(config.vio_debug){
      std::cout<<"detected points: "<<kd.corners.size()<<std::endl;
      std::cout<<"remain points(remove out-of-boundry ones): "<<kd.corners.size() - kp_to_remove.size()<<std::endl<<std::endl;
    }

    for (size_t i = 0; i < kd.corners.size(); i++) {
      if(kp_to_remove.count(i)) continue;
      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      transforms->observations.at(0)[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    if (calib.intrinsics.size() > 1) {
      if(config.vio_debug){
          std::cout<<"track from left frame to right"<<std::endl;
        }
      // Yu: apply cameara calibration here!!!!
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1, true);

      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }
      if(config.vio_debug){
        std::cout<<"cam0 newly detected points: "<< transforms->observations.at(0).size()
        <<" cam1 tracked points from cam0: "<<transforms->observations.at(1).size()<<" track ratio: "
        << float(transforms->observations.at(1).size())/transforms->observations.at(0).size()<<std::endl;
      }
    }
  }

  void filterPoints() {
    if (calib.intrinsics.size() < 2) return;

    std::set<KeypointId> lm_to_remove;
    std::set<KeypointId> lm_to_remove_left; //Yu remove feature in left image as well if valid the unproject creteria


    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto& kv : transforms->observations.at(1)) {
      // hm: find the same keypoint ID in both camera
      auto it = transforms->observations.at(0).find(kv.first);

      if (it != transforms->observations.at(0).end()) { // hm: found 
        proj0.emplace_back(it->second.translation());
        proj1.emplace_back(kv.second.translation());
        kpid.emplace_back(kv.first);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success); // hm: unproject all image domain points in cam0
    calib.intrinsics[1].unproject(proj1, p3d1, p3d1_success); // hm: unproject all image domain points in cam1

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            // hm: Essential Matrix - for homogeneous normalized image coordinates 
            // hm: E only the topleft 3x3 part is used
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if(!p3d0_success[i]) lm_to_remove_left.emplace(kpid[i]);
    }


    if(config.vio_debug){
      std::cout<<"kp: "<<kpid.size()<<std::endl;
      std::cout<<"lm_to_remove_left: "<<lm_to_remove_left.size()<<std::endl;
      std::cout<<"lm_to_remove_right: "<<lm_to_remove.size()<<std::endl;
    }

    for (int id : lm_to_remove) {
      transforms->observations.at(1).erase(id);
    }

    for (int id : lm_to_remove_left) {
      transforms->observations.at(0).erase(id);
    }

    if (lm_to_remove.size())
      std::cout<<"remove " <<lm_to_remove.size() <<" points valid epipolar constrain, remove ratio: "
        << float(lm_to_remove.size() )/transforms->observations.at(1).size()<<std::endl;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int64_t t_ns;

  size_t frame_counter;

  KeypointId last_keypoint_id;

  VioConfig config;
  basalt::Calibration<Scalar> calib;

  OpticalFlowResult::Ptr transforms;
  std::shared_ptr<std::vector<basalt::ManagedImagePyr<u_int16_t>>> old_pyramid,
      pyramid;

  Matrix4 E;

  std::shared_ptr<std::thread> processing_thread;
};

}  // namespace basalt

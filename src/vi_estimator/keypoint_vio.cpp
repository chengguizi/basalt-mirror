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

#include <basalt/utils/assert.h>
#include <basalt/vi_estimator/keypoint_vio.h>

#include <basalt/optimization/accumulator.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <chrono>

namespace basalt {

KeypointVioEstimator::KeypointVioEstimator(
    const Eigen::Vector3d& g, const basalt::Calibration<double>& calib,
    const VioConfig& config)
    : take_kf(true),
      frames_after_kf(0),
      g(g),
      initialized(false),
      config(config),
      lambda(config.vio_lm_lambda_min),
      min_lambda(config.vio_lm_lambda_min),
      max_lambda(config.vio_lm_lambda_max),
      lambda_vee(2) {
  this->obs_std_dev = config.vio_obs_std_dev;
  this->huber_thresh = config.vio_obs_huber_thresh;
  this->calib = calib;

  // Setup marginalization
  marg_H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  marg_b.setZero(POSE_VEL_BIAS_SIZE);

  // prior on position
  // hm: this is used to fix the initial position close to the origin?
  marg_H.diagonal().head<3>().setConstant(config.vio_init_pose_weight);
  // prior on yaw
  marg_H(5, 5) = config.vio_init_pose_weight;

  // small prior to avoid jumps in bias
  marg_H.diagonal().segment<3>(9).array() = config.vio_init_ba_weight;
  marg_H.diagonal().segment<3>(12).array() = config.vio_init_bg_weight;

  std::cout << "marg_H\n" << marg_H << std::endl;
  std::cout<<"feature_match_show: "<<config.feature_match_show<<std::endl;  

  gyro_bias_weight = calib.gyro_bias_std.array().square().inverse();
  accel_bias_weight = calib.accel_bias_std.array().square().inverse();

  max_states = config.vio_max_states;
  max_kfs = config.vio_max_kfs;

  opt_started = false;

  vision_data_queue.set_capacity(100);
  imu_data_queue.set_capacity(1000);

  std::cout.precision(15);
}

void KeypointVioEstimator::initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                                      const Eigen::Vector3d& vel_w_i,
                                      const Eigen::Vector3d& bg,
                                      const Eigen::Vector3d& ba) {
  initialized = true;
  T_w_i_init = T_w_i;

  last_state_t_ns = t_ns;
  // imu_meas[t_ns] = IntegratedImuMeasurement(t_ns, bg, ba);
  frame_states[t_ns] =
      PoseVelBiasStateWithLin(t_ns, T_w_i, vel_w_i, bg, ba, true);

  // hm: this marg_order entry correspond to the frame_states added above, the first full state, with no keyframes yet
  marg_order.abs_order_map[t_ns] = std::make_pair(0, POSE_VEL_BIAS_SIZE);
  marg_order.total_size = POSE_VEL_BIAS_SIZE;
  marg_order.items = 1;

  initialize(bg, ba);
}

void KeypointVioEstimator::initialize(const Eigen::Vector3d& bg,
                                      const Eigen::Vector3d& ba) {
  auto proc_func = [&, bg, ba] {
    OpticalFlowResult::Ptr prev_frame, curr_frame;
    IntegratedImuMeasurement::Ptr meas;

    const Eigen::Vector3d accel_cov =
        calib.dicrete_time_accel_noise_std().array().square();
    const Eigen::Vector3d gyro_cov =
        calib.dicrete_time_gyro_noise_std().array().square();

    ImuData::Ptr data;
    imu_data_queue.pop(data);
    data->accel = calib.calib_accel_bias.getCalibrated(data->accel).transpose();
    data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro).transpose();

    while (true) {
      if(config.vio_debug){std::cout<<"opt_flow queue size: "<<vision_data_queue.size()<<std::endl;}
      vision_data_queue.pop(curr_frame);
      int skipped_image = 0;
      if (config.vio_enforce_realtime) {
        // drop current frame if another frame is already in the queue.
        while (vision_data_queue.try_pop(curr_frame)) {skipped_image++;}
        if(skipped_image)
          std::cerr<< "[Warning] skipped opt flow size: "<<skipped_image<<std::endl;
      }
      if (!curr_frame.get()) {
        break;
      }

      // Correct camera time offset
      curr_frame->t_ns += calib.cam_time_offset_ns;
      int imu_num{0};
      int skipped_imu{0};

      if(config.vio_debug) std::cout << "got frame data at time " << double(curr_frame->t_ns * 1.e-9) << std::endl;
      if (!initialized) {
         // hm: ensure frame arrive after first data
        if (curr_frame->t_ns < data->t_ns)
          continue;

        // hm: data is the pointer to IMU measurement
        // hm: throw away old imu data before first frame
        while (data->t_ns < curr_frame->t_ns) {
          imu_data_queue.pop(data);
          if (!data.get()) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
          // std::cout << "Skipping IMU data.." << std::endl;
        }
        // hm: after the while loop, only a single point data from the IMU is kept. This can be quite bad

        // hm: initialise the velocity as zero, regardless
        Eigen::Vector3d vel_w_i_init;
        vel_w_i_init.setZero();

        // hm: FromTwoVectors will return the rotation in quaternion, to make the first vector the same as the second
        // hm: This basically a change of basis from body frame to global frame
        // hm: This does not give any treatment for yaw
        Eigen::Quaterniond q_g_b(Eigen::Quaterniond::FromTwoVectors(
            data->accel, Eigen::Vector3d::UnitZ()));
        // T_w_i_init.setQuaternion(Eigen::Quaterniond::FromTwoVectors(
            // data->accel, Eigen::Vector3d::UnitZ()));
          Eigen::Matrix<double, 3, 3> M_w_i;  //Yu: rotation matrix from imu to world
          M_w_i<< 0, -1, 0,
                  1, 0, 0,
                  0, 0, 1;

        Eigen::Quaterniond q_w_i(M_w_i);
        std::cout<<"q_w_i: "<<q_w_i.w()<<", "<<q_w_i.x()<<", "<<q_w_i.y()<<", "<<q_w_i.z()<<std::endl;
        T_w_i_init.setQuaternion(q_w_i*q_g_b);

        last_state_t_ns = curr_frame->t_ns;
        // imu_meas[last_state_t_ns] =
        //     IntegratedImuMeasurement(last_state_t_ns, bg, ba);
        frame_states[last_state_t_ns] = PoseVelBiasStateWithLin(
            last_state_t_ns, T_w_i_init, vel_w_i_init, bg, ba, true);

        // hm: correspond to the first frame_states added above
        marg_order.abs_order_map[last_state_t_ns] =
            std::make_pair(0, POSE_VEL_BIAS_SIZE);
        marg_order.total_size = POSE_VEL_BIAS_SIZE;
        marg_order.items = 1;

        std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
        std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;
        std::cout << "vel_w_i " << vel_w_i_init.transpose() << std::endl;

        initialized = true;
        BASALT_ASSERT(imu_meas.size() == 0);
      }

      if (prev_frame) {
        // preintegrate measurements

        // hm: retrieve the last available state in the buffer
        auto last_state = frame_states.at(last_state_t_ns);

        // hm: mark the pre-integration start time as previous frame's time
        meas.reset(new IntegratedImuMeasurement(
            prev_frame->t_ns, last_state.getState().bias_gyro,
            last_state.getState().bias_accel));
        
        // if (config.vio_debug) {
        //   std::cout<<"gyro bias: "<<last_state.getState().bias_gyro.transpose()<<std::endl;
        //   std::cout<<"accel bias: "<<last_state.getState().bias_accel.transpose()<<std::endl<<std::endl;
        // }

        // hm: data is the pointer to IMU measurement, throw away all imu prior to the previous frame

        while (data->t_ns <= prev_frame->t_ns) {
          imu_data_queue.pop(data);
          if (!data.get()) break;
          // hm: to be used in the first run of the while loop below
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
          skipped_imu++;
          if(config.vio_debug) std::cout << "popped imu data at time " << double(data->t_ns * 1.e-9) << std::endl;
        }

        // hm: starting from the first imu reading AFTER previous frame time, integrated until the last imu reading before current frame
        // hm: this may block, due to .pop()
        auto pre_imu_time = prev_frame->t_ns;
        while (data->t_ns <= curr_frame->t_ns) {
          if (config.vio_debug) {
            std::cout<<"time diff"<<imu_num<<" btw imu frame: "<<double(data->t_ns * 1.e-9)<< "-" <<double(pre_imu_time * 1.e-9)<<" = "<<double((data->t_ns - pre_imu_time) * 1.e-9)<<std::endl;
          }
          pre_imu_time = data->t_ns;

          // hm: noise covariance is from config, fixed variance
          meas->integrate(*data, accel_cov, gyro_cov);
          // hm: pop for the next round of while loop
          imu_data_queue.pop(data);

          // hm: if the pipe ends with null pointer
          if (!data.get()) break;
          if(config.vio_debug) std::cout << "popped imu data at time " << double(data->t_ns * 1.e-9) << std::endl;
          // hm: obtain the correction scale and bias, and store it back to imu data
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
          imu_num++;
        }

        // hm: this is the case where the last imu time has a gap to the current frame
        if (meas->get_start_t_ns() + meas->get_dt_ns() < curr_frame->t_ns) {

          if (config.vio_debug) {
            std::cout<<"time diff btw imu frame (to current frame): "<<double(curr_frame->t_ns * 1.e-9)<< "-" <<double((meas->get_dt_ns() + meas->get_start_t_ns()) * 1.e-9)<<" = "<<double((curr_frame->t_ns - meas->get_dt_ns() - meas->get_start_t_ns()) * 1.e-9) << std::endl;
          }

          // hm: maximum 20ms of IMU time modification is allowed
          BASALT_ASSERT(curr_frame->t_ns - (meas->get_start_t_ns() + meas->get_dt_ns()) < 20e6);

          int64_t tmp = data->t_ns;
          data->t_ns = curr_frame->t_ns;
          meas->integrate(*data, accel_cov, gyro_cov);
          data->t_ns = tmp;
          imu_num++;

        }

        if (config.vio_debug) {
          std::cout<<"time between frames: "<<double(curr_frame->t_ns * 1.e-9)<< "-" <<double(prev_frame->t_ns * 1.e-9)<<" = "<<double((curr_frame->t_ns - prev_frame->t_ns) * 1.e-9)<<std::endl;
          std::cout<<"imu num used for preintegration: "<<imu_num<<std::endl;
          std::cout<<"skipped imu between frames: "<<skipped_imu<<std::endl;
          std::cout<<"imu buffer size: "<<imu_data_queue.size()<<std::endl;
          std::cout<<"current latest imu timestamp: "<<double(data->t_ns * 1.e-9) <<std::endl;

        }
      }

      // hm: pass the optical flow result, and the pre-integration result
      measure(curr_frame, meas);
      prev_frame = curr_frame;
    }

    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_marg_queue) out_marg_queue->push(nullptr);
    if (out_state_queue) out_state_queue->push(nullptr);

    finished = true;

    std::cout << "Finished VIOFilter " << std::endl;
  };

  processing_thread.reset(new std::thread(proc_func));
}

void KeypointVioEstimator::addIMUToQueue(const ImuData::Ptr& data) {
  imu_data_queue.emplace(data);
}

void KeypointVioEstimator::addVisionToQueue(
    const OpticalFlowResult::Ptr& data) {

  if(!vision_data_queue.try_push(data)){
    std::cout<<"visiond data queue is full: "<<vision_data_queue.size()<<std::endl;
    abort();
  }
}

// hm: measure takes in the current frame flow result, with the pre-integration results (previous frame -> current frame)
bool KeypointVioEstimator::measure(const OpticalFlowResult::Ptr& opt_flow_meas,
                                   const IntegratedImuMeasurement::Ptr& meas) {

  //// Process IMU readings
  if (meas.get()) {
    // hm: last_state_t_ns is the index of the latest frame state (which is the previous frame), pos/vel/bias
    BASALT_ASSERT(frame_states[last_state_t_ns].getState().t_ns ==
                  meas->get_start_t_ns());
    BASALT_ASSERT(opt_flow_meas->t_ns ==
                  meas->get_dt_ns() + meas->get_start_t_ns());

    // hm: this is to be updated by predictState() immediately below
    PoseVelBiasState next_state = frame_states.at(last_state_t_ns).getState();

    // hm: here g is a static constant defined by imu_type.h
    // hm: this performs the pre-integration, based on a given starting state, and the pre-integrated imu measurement
    meas->predictState(frame_states.at(last_state_t_ns).getState(), g,
                       next_state);
    
    if (config.vio_debug) {
      Eigen::AngleAxisd angleAxis(next_state.T_w_i.unit_quaternion());
      std::cout<< "vel_w_i: "<<next_state.vel_w_i.transpose()<<std::endl;
      std::cout << "T: " << next_state.T_w_i.translation().transpose()<< std::endl;
      std::cout << "AxisAngle: " << angleAxis.angle() * 57.3 << ", "<<angleAxis.axis().transpose()<< std::endl<<std::endl;
   
      PoseVelState delta_state = meas->getDeltaState();
      Eigen::AngleAxisd angleAxisDelta(delta_state.T_w_i.unit_quaternion());
      std::cout<< "delta_state_vel_w_i: "<<delta_state.vel_w_i.transpose()<<std::endl;
      std::cout << "delta_state_T: " << delta_state.T_w_i.translation().transpose()<< std::endl;
      std::cout << "delta_state_AxisAngle: " << angleAxisDelta.angle() * 57.3 << ", "<<angleAxisDelta.axis().transpose()<< std::endl<<std::endl;
    }

    // hm: last_state_t_ns now stores the current frame timestamp, instead of the previous frame
    last_state_t_ns = opt_flow_meas->t_ns;
    // hm: next_state stores the current frame pose estimation, given pre-integrated imu measurement
    next_state.t_ns = opt_flow_meas->t_ns;

    // hm: now we have the predicted state added into the frame state buffer
    // hm: this is a full state, before taking the measurement
    frame_states[last_state_t_ns] = next_state;

    // hm: we also update the imu measurement buffer
    BASALT_ASSERT(imu_meas.count(meas->get_start_t_ns()) == 0); // hm: assert we are not overriding any measurement
    imu_meas[meas->get_start_t_ns()] = *meas;
  }
  else{
    std::cerr<<"skip imu measurement update once"<<std::endl;
    // hm: on the first frame, imu integration is not ready yet, so no meas pointer yet.
    if (imu_meas.size() > 0)
      throw std::runtime_error("abort");
  }

  // save results
  // hm: add optical flow results in, good for optimisation later
  // hm: to be removed with marginalisation in due time
  prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

  // Make new residual for existing keypoints
  int connected0 = 0;
  std::map<int64_t, int> num_points_connected;
  std::unordered_set<int> unconnected_obs0;

  // hm: going through each camera to process each camera's observations of optical flow, current frame
  for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
    // hm: frame_id, camera_id
    TimeCamId tcid_target(opt_flow_meas->t_ns, i);
    // Yu: add observations to landmark
    for (const auto& kv_obs : opt_flow_meas->observations[i]) {
      int kpt_id = kv_obs.first;

      // hm: landmark map, from the bundle adjustment 
      if (lmdb.landmarkExists(kpt_id)) {
        // hm: obtain the keyframe that hosts the keypoint
        const TimeCamId& tcid_host = lmdb.getLandmark(kpt_id).kf_id;

        KeypointObservation kobs;
        kobs.kpt_id = kpt_id;
        // hm: note, only translation is used here, not rotation/reflection/shearing
        kobs.pos = kv_obs.second.translation().cast<double>();

        lmdb.addObservation(tcid_target, kobs);
        // obs[tcid_host][tcid_target].push_back(kobs);


        // hm: return either 0 or 1, if 0 then initialise the key
        // hm: num_points_connected tracks, for each key frame, the number of optical flow observations (from the current frame) belong to it
        if (num_points_connected.count(tcid_host.frame_id) == 0) {
          num_points_connected[tcid_host.frame_id] = 0;
        }
        num_points_connected[tcid_host.frame_id]++;

        // hm: if it is the first camera, then it is in all cameras
        // hm: only observations linked to a keyframe's landmark is consider connected
        if (i == 0) connected0++;
      } else {
        // hm: i==0 checks ensures that one keypoint is added at most once to the set
        if (i == 0) {
          unconnected_obs0.emplace(kpt_id);
        }
      }
      
    }

    if (config.vio_debug) {
      std::cout << "cam " << i << " observation size = " <<  opt_flow_meas->observations[i].size() << std::endl;
      std::cout << "connected0 = " << connected0 << std::endl;
      std::cout << "No. of landmarks in the database: " <<  lmdb.numLandmarks() << std::endl;
    }
  }

  // hm: check if keyframe is needed
  // hm: criteria 1: vio_new_kf_keypoints_thresh, the ratio between landmarked observations and the total observations
  // hm: criteria 2: vio_min_frames_after_kf, having minimum frames in between
  if ( (lmdb.numLandmarks() < 20 || lmdb.numLandmarks() / unconnected_obs0.size() < 0.05 || double(connected0) / lmdb.numLandmarks() < config.vio_new_kf_keypoints_thresh)
        && (frames_after_kf > config.vio_min_frames_after_kf))
    take_kf = true;


  //// hm: step in initialising a keyframes:
  // hm: 1. the key frame id is defined as the current frame nanosecond timestamp
  // hm: 2. each landmark id is defined as corresponding keypoint id (unique)
  // hm: 3. collect all observations for each landmark id, put in the variable `kp_obs`. both in time, and in multi camera view
  // hm: 4.0 then iterate over all observations, doing triangulation verification: p0 is the first camera observation (current frame), p1 is the one which iterates (from the oldest time to most current)
  // hm: 4.1 for each p1, obtain the relative camera pose transformation to p0, T_0_1
  // hm: 4.2 translation has a threshold vio_min_triangulation_dist, to ensure proper triangulation
  // hm: 4.3 triangulation correctness is NOT checked

  if (take_kf) {
    if (config.vio_debug)
      std::cout << "Taking keyframe now, unconnected_obs0.size() = " << unconnected_obs0.size() << std::endl;

    // Triangulate new points from stereo and make keyframe for camera 0
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);

    // hm: 0 means first camera
    TimeCamId tcidl(opt_flow_meas->t_ns, 0);

    BASALT_ASSERT(last_state_t_ns == opt_flow_meas->t_ns);

    int num_points_added = 0;
    // hm: loop over unconnected keypoint ids
    for (int lm_id : unconnected_obs0) {
      // Find all observations
      // hm: for a given keypoint (id = lm_id)
      std::map<TimeCamId, KeypointObservation> kp_obs;

      // hm: construct kp_obs
      // hm: kv iterates over time
      for (const auto& kv : prev_opt_flow_res) {
        // hm: k = camera id, observations.size() is the number of cameras
        for (size_t k = 0; k < kv.second->observations.size(); k++) {
          // hm: at the end, kp_obs stores the observations for each view in each timestamp
          auto it = kv.second->observations[k].find(lm_id);
          if (it != kv.second->observations[k].end()) {
            // hm: it now is the observation
            // hm: time(keyframe id), and camera id
            TimeCamId tcido(kv.first, k);

            KeypointObservation kobs;
            kobs.kpt_id = lm_id;
            kobs.pos = it->second.translation().cast<double>();

            // obs[tcidl][tcido].push_back(kobs);
            kp_obs[tcido] = kobs;
          }
        }
      }

      // hm: now kp_obs stores all observations pointing to the same landmark, over time and over camera views

      // triangulate
      bool valid_kp = false;
      // hm: config vio_min_triangulation_dist
      const double min_triang_distance2 =
          config.vio_min_triangulation_dist * config.vio_min_triangulation_dist;
      // hm: loop over all cameras, for each observation of the SAME keypoint
      for (const auto& kv_obs : kp_obs) {
        //Yu: break once we find a valid 3d point between this and one of previous observations 
        if (valid_kp) break;     
        TimeCamId tcido = kv_obs.first;

        const Eigen::Vector2d p0 = opt_flow_meas->observations.at(0)
                                       .at(lm_id)
                                       .translation()
                                       .cast<double>();
        const Eigen::Vector2d p1 = prev_opt_flow_res[tcido.frame_id]
                                       ->observations[tcido.cam_id]
                                       .at(lm_id)
                                       .translation()
                                       .cast<double>();

        Eigen::Vector4d p0_3d, p1_3d;
        bool valid1 = calib.intrinsics[0].unproject(p0, p0_3d);
        bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
        if (!valid1 || !valid2) continue;

        // hm: the pose would be far at first, as it is sorted by time and then by camera views
        Sophus::SE3d T_i0_i1 =
            getPoseStateWithLin(tcidl.frame_id).getPose().inverse() *
            getPoseStateWithLin(tcido.frame_id).getPose();
        // hm: camera to camera transformation
        Sophus::SE3d T_0_1 =
            calib.T_i_c[0].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

        // hm: require distance between the cameras to be large enough
        if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

        Eigen::Vector4d p0_triangulated =
            triangulate(p0_3d.head<3>(), p1_3d.head<3>(), T_0_1);

        // if(config.vio_debug){
        //   std::cout<< "lm_id: " << lm_id << ", p0_triangulated: "<<p0_triangulated.transpose()<<std::endl;
        // }

        // hm: distance criteria: the homogeneous part is reasonable
        if (p0_triangulated.array().isFinite().all() &&
            p0_triangulated[3] > 0 && p0_triangulated[3] < 3.0) {
          // hm: defined in the landmark_database
          KeypointPosition kpt_pos;
          kpt_pos.kf_id = tcidl;
          // hm: representation of 3d direction, using 2 numbers
          kpt_pos.dir = StereographicParam<double>::project(p0_triangulated);
          // hm: inverse distance
          kpt_pos.id = p0_triangulated[3];
          lmdb.addLandmark(lm_id, kpt_pos);

          num_points_added++;
          valid_kp = true;
        }
      }
      // Yu:add all the observations to the newly added point
      // if(config.vio_debug){
      //     std::cout<<"valid_kp: "<<valid_kp<<std::endl;
      // }
      if (valid_kp) {
        for (const auto& kv_obs : kp_obs) {
          lmdb.addObservation(kv_obs.first, kv_obs.second);
        }
      }
    }

    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }
  // hm: end of taking keyframe

  optimize();
  marginalize(num_points_connected);

  if (out_state_queue) {
    PoseVelBiasStateWithLin p = frame_states.at(last_state_t_ns);

    PoseVelBiasState::Ptr data(new PoseVelBiasState(p.getState()));

    out_state_queue->push(data);

    // hm: debug bias
    std::cout << "bias_accel " << data->bias_accel.transpose() << std::endl;
    std::cout << "bias_gyro " << data->bias_gyro.transpose() << std::endl;
  }

  if (out_vis_queue) {
    VioVisualizationData::Ptr data(new VioVisualizationData);

    data->t_ns = last_state_t_ns;

    for (const auto& kv : frame_states) {
      data->states.emplace_back(kv.second.getState().T_w_i);
    }

    for (const auto& kv : frame_poses) {
      data->frames.emplace_back(kv.second.getPose());
    }

    // hm: obtain landmarks' 3d points in world frame
    get_current_points(data->points, data->point_ids);

    // hm: resize to the number of cameras
    data->projections.resize(opt_flow_meas->observations.size());

    // hm: projections are split into cameras, for each camera, it is a vector of coordinates (4th number is LANDMARK ID)
    // hm: NOTE, only landmarks in the database are projected
    computeProjections(data->projections);

    data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

    out_vis_queue->push(data);
  }

  last_processed_t_ns = last_state_t_ns;
  if(config.vio_debug){
    std::cout<<"numLandmarks: "<<lmdb.numLandmarks()<<std::endl;
  }

  return true;
}

void KeypointVioEstimator::checkMargNullspace() const {
  checkNullspace(marg_H, marg_b, marg_order, frame_states, frame_poses);
}

void KeypointVioEstimator::marginalize(
    const std::map<int64_t, int>& num_points_connected) {
  // hm: a flag to indicate that optimistion step has been started, after a skip of frames at the very beginning
  if (!opt_started) return;

  // hm: condition 1, keyframe size exceed (frame_poses, are obtained when states_to_marg_vel_bias is true, where vel and bias are removed -> keyframe)
  // hm: condition 2, total states (excluding keyframes) exceed max states
  if (frame_poses.size() > max_kfs || frame_states.size() >= max_states) {
    // Marginalize

    // hm: 1 less than the max_states
    const int states_to_remove = frame_states.size() - max_states + 1;

    auto it = frame_states.cbegin();
    for (int i = 0; i < states_to_remove; i++) it++;
    int64_t last_state_to_marg = it->first;

    AbsOrderMap aom;

    // remove all frame_poses that are not kfs
    // hm: all poses are supposed to be key frames anyway?
    std::set<int64_t> poses_to_marg;
    for (const auto& kv : frame_poses) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

      // hm: kf_ids stores the keyframes' timestamp as a set
      // hm: if it is not a keyframe, then to be marginalised. but why?
      // hm: maybe it is determined that it should no longer be a keyframe, later on in its lifetime?
      if (kf_ids.count(kv.first) == 0) poses_to_marg.emplace(kv.first);

      // Check that we have the same order as marginalization
      BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                    aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_SIZE;
      aom.items++;
    }

    // hm: here, we are trying to remove all (non-keyframe) states prior to the last_state_to_marg
    // hm: WHY? what if when kv.first == last_state_to_marg?
    std::set<int64_t> states_to_marg_vel_bias;
    std::set<int64_t> states_to_marg_all;
    for (const auto& kv : frame_states) {
      if (kv.first > last_state_to_marg) break;

      if (kv.first != last_state_to_marg) {
        // hm: if it is a keyframe, marginalise its velocity and bias, keep the T and landmarks?
        if (kf_ids.count(kv.first) > 0) {
          states_to_marg_vel_bias.emplace(kv.first);
        } else {
          states_to_marg_all.emplace(kv.first);
        }
      }

      aom.abs_order_map[kv.first] =
          std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

      // Check that we have the same order as marginalization
      if (aom.items < marg_order.abs_order_map.size())
        BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                      aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_VEL_BIAS_SIZE;
      aom.items++;
    }

    //// hm: determining which key frames to be marginalised
    auto kf_ids_all = kf_ids;
    std::set<int64_t> kfs_to_marg;
    // hm: there are key frames to be marginalised, also upon condition that maximum key frame is reached, AND there are new keyframes to be marginalised
    while (kf_ids.size() > max_kfs && !states_to_marg_vel_bias.empty()) {
      int64_t id_to_marg = -1;

      // hm: to marginalised little-overlapping key frames
      {
        // hm: convert set to vector data structure
        std::vector<int64_t> ids;
        for (int64_t id : kf_ids) {
          ids.push_back(id);
        }

        for (size_t i = 0; i < ids.size() - 2; i++) {
          // hm: num_points_connected: number of observations at this frame
          // hm: num_points_kf: as a key frame, how many observations are added
          // hm: hence, when the key frame contains too few keypoints observed in the current frame => to marginalise
          if (num_points_connected.count(ids[i]) == 0 ||
              (num_points_connected.at(ids[i]) / num_points_kf.at(ids[i]) <
               0.05)) {
            id_to_marg = ids[i];
            break;
          }
        }
      }

      // hm: if all sufficiently overlapping, choose the one closest to the last key frame
      if (id_to_marg < 0) {
        std::vector<int64_t> ids;
        for (int64_t id : kf_ids) {
          ids.push_back(id);
        }

        int64_t last_kf = *kf_ids.crbegin();
        double min_score = std::numeric_limits<double>::max();
        int64_t min_score_id = -1;

        for (size_t i = 0; i < ids.size() - 2; i++) {
          double denom = 0;
          // hm: demorminator: 'average' similarity between all key frames
          for (size_t j = 0; j < ids.size() - 2; j++) {
            denom += 1 / ((frame_poses.at(ids[i]).getPose().translation() -
                           frame_poses.at(ids[j]).getPose().translation())
                              .norm() +
                          1e-5);
          }

          double score =
              std::sqrt(
                  (frame_poses.at(ids[i]).getPose().translation() -
                   frame_states.at(last_kf).getState().T_w_i.translation())
                      .norm()) *
              denom;

          if (score < min_score) {
            min_score_id = ids[i];
            min_score = score;
          }
        }

        id_to_marg = min_score_id;
      }

      kfs_to_marg.emplace(id_to_marg);
      // hm: poses_to_marg.emplace is called else where too
      poses_to_marg.emplace(id_to_marg);

      kf_ids.erase(id_to_marg);
    }

    //    std::cout << "marg order" << std::endl;
    //    aom.print_order();

    //    std::cout << "marg prior order" << std::endl;
    //    marg_order.print_order();

    if (config.vio_debug) {
      std::cout << "states_to_remove " << states_to_remove << std::endl;
      std::cout << "poses_to_marg.size() " << poses_to_marg.size() << std::endl;
      std::cout << "states_to_marg.size() " << states_to_marg_all.size()
                << std::endl;
      std::cout << "state_to_marg_vel_bias.size() "
                << states_to_marg_vel_bias.size() << std::endl;
      std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
    }

    size_t asize = aom.total_size;

    double marg_prior_error;
    double imu_error, bg_error, ba_error;

    DenseAccumulator accum;
    accum.reset(asize);

    {
      // Linearize points

      Eigen::aligned_map<
          TimeCamId, Eigen::aligned_map<
                         TimeCamId, Eigen::aligned_vector<KeypointObservation>>>
          obs_to_lin;

      for (auto it = lmdb.getObservations().cbegin();
           it != lmdb.getObservations().cend();) {
        if (kfs_to_marg.count(it->first.frame_id) > 0) {
          for (auto it2 = it->second.cbegin(); it2 != it->second.cend();
               ++it2) {
            if (it2->first.frame_id <= last_state_to_marg)
              obs_to_lin[it->first].emplace(*it2);
          }
        }
        ++it;
      }

      double rld_error;
      Eigen::aligned_vector<RelLinData> rld_vec;

      linearizeHelper(rld_vec, obs_to_lin, rld_error);

      for (auto& rld : rld_vec) {
        rld.invert_keypoint_hessians();

        Eigen::MatrixXd rel_H;
        Eigen::VectorXd rel_b;
        linearizeRel(rld, rel_H, rel_b);

        linearizeAbs(rel_H, rel_b, rld, aom, accum);
      }
    }

    linearizeAbsIMU(aom, accum.getH(), accum.getB(), imu_error, bg_error,
                    ba_error, frame_states, imu_meas, gyro_bias_weight,
                    accel_bias_weight, g);
    linearizeMargPrior(marg_order, marg_H, marg_b, aom, accum.getH(),
                       accum.getB(), marg_prior_error);

    // Save marginalization prior
    if (out_marg_queue && !kfs_to_marg.empty()) {
      // int64_t kf_id = *kfs_to_marg.begin();

      {
        MargData::Ptr m(new MargData);
        m->aom = aom;
        m->abs_H = accum.getH();
        m->abs_b = accum.getB();
        m->frame_poses = frame_poses;
        m->frame_states = frame_states;
        m->kfs_all = kf_ids_all;
        m->kfs_to_marg = kfs_to_marg;
        m->use_imu = true;

        for (int64_t t : m->kfs_all) {
          m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
        }

        out_marg_queue->push(m);
      }
    }

    std::set<int> idx_to_keep, idx_to_marg;
    for (const auto& kv : aom.abs_order_map) {
      if (kv.second.second == POSE_SIZE) {
        int start_idx = kv.second.first;
        if (poses_to_marg.count(kv.first) == 0) {
          for (size_t i = 0; i < POSE_SIZE; i++)
            idx_to_keep.emplace(start_idx + i);
        } else {
          for (size_t i = 0; i < POSE_SIZE; i++)
            idx_to_marg.emplace(start_idx + i);
        }
      } else {
        BASALT_ASSERT(kv.second.second == POSE_VEL_BIAS_SIZE);
        // state
        int start_idx = kv.second.first;
        if (states_to_marg_all.count(kv.first) > 0) {
          for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
            idx_to_marg.emplace(start_idx + i);
        } else if (states_to_marg_vel_bias.count(kv.first) > 0) {
          for (size_t i = 0; i < POSE_SIZE; i++)
            idx_to_keep.emplace(start_idx + i);
          for (size_t i = POSE_SIZE; i < POSE_VEL_BIAS_SIZE; i++)
            idx_to_marg.emplace(start_idx + i);
        } else {
          BASALT_ASSERT(kv.first == last_state_to_marg);
          for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
            idx_to_keep.emplace(start_idx + i);
        }
      }
    }

    if (config.vio_debug) {
      std::cout << "keeping " << idx_to_keep.size() << " marg "
                << idx_to_marg.size() << " total " << asize << std::endl;
      std::cout << "last_state_to_marg " << last_state_to_marg
                << " frame_poses " << frame_poses.size() << " frame_states "
                << frame_states.size() << std::endl;
    }

    Eigen::MatrixXd marg_H_new;
    Eigen::VectorXd marg_b_new;
    marginalizeHelper(accum.getH(), accum.getB(), idx_to_keep, idx_to_marg,
                      marg_H_new, marg_b_new);

    {
      BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() ==
                    false);
      frame_states.at(last_state_to_marg).setLinTrue();
    }

    for (const int64_t id : states_to_marg_all) {
      frame_states.erase(id);
      imu_meas.erase(id);
      prev_opt_flow_res.erase(id);
    }

    for (const int64_t id : states_to_marg_vel_bias) {
      const PoseVelBiasStateWithLin& state = frame_states.at(id);
      PoseStateWithLin pose(state);

      frame_poses[id] = pose;
      frame_states.erase(id);
      imu_meas.erase(id);
    }

    for (const int64_t id : poses_to_marg) {
      frame_poses.erase(id);
      prev_opt_flow_res.erase(id);
    }

    lmdb.removeKeyframes(kfs_to_marg, poses_to_marg, states_to_marg_all);

    // hm: calculating the new states in order, it contains all marginalised pose states, and one latest full state
    AbsOrderMap marg_order_new;

    for (const auto& kv : frame_poses) {
      marg_order_new.abs_order_map[kv.first] =
          std::make_pair(marg_order_new.total_size, POSE_SIZE);

      marg_order_new.total_size += POSE_SIZE;
      marg_order_new.items++;
    }

    {
      marg_order_new.abs_order_map[last_state_to_marg] =
          std::make_pair(marg_order_new.total_size, POSE_VEL_BIAS_SIZE);
      marg_order_new.total_size += POSE_VEL_BIAS_SIZE;
      marg_order_new.items++;
    }

    marg_H = marg_H_new;
    marg_b = marg_b_new;
    marg_order = marg_order_new;

    BASALT_ASSERT(size_t(marg_H.cols()) == marg_order.total_size);

    Eigen::VectorXd delta;
    computeDelta(marg_order, delta);
    marg_b -= marg_H * delta;

    if (config.vio_debug) {
      std::cout << "marginalizaon done!!" << std::endl;

      std::cout << "======== Marg nullspace ==========" << std::endl;
      checkMargNullspace();
      std::cout << "=================================" << std::endl;
    }

    //    std::cout << "new marg prior order" << std::endl;
    //    marg_order.print_order();
  }
}

void KeypointVioEstimator::optimize() {
  if (config.vio_debug) {
    std::cout << "===============optimize()==================" << std::endl;
  }

  // hm: optimisation started only after a hardcoded 5 frames
  if (opt_started || frame_states.size() > 4) {
    // Optimize
    opt_started = true;

    // hm: AbsOrderMap is std::map, ordered key-value pairs, normally implemented as red-black tree (sorted by key)
    // hm: query takes log(n)
    // hm: this is just to store all states (either full or marginalised) in order
    AbsOrderMap aom;

    // hm: sequentially store all the frame_poses (key frames) in the aom
    for (const auto& kv : frame_poses) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

      // Check that we have the same order as marginalization
      BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                    aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_SIZE;
      aom.items++;
    }

    // hm: then sequentially store all frame_states (full states) in aom
    for (const auto& kv : frame_states) {
      aom.abs_order_map[kv.first] =
          std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

      // Check that we have the same order as marginalization
      if (aom.items < marg_order.abs_order_map.size())
        BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                      aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_VEL_BIAS_SIZE;
      aom.items++;
    }

    //    std::cout << "opt order" << std::endl;
    //    aom.print_order();

    //    std::cout << "marg prior order" << std::endl;
    //    marg_order.print_order();

    // hm: doing optimisation for vio_max_iterations times, unless converged
    // hm: Note: at vio_filter_iteration iteration, additional filterOutliers is done
    for (int iter = 0; iter < config.vio_max_iterations; iter++) {
      auto t1 = std::chrono::high_resolution_clock::now();

      // hm: sum of all error from all landmarks
      double rld_error;
      // hm: this vector store all the partial derivative of the error loss of a landmark, indexing the landmark
      // hm: includes different H matrix for each landmark's observations, respect to both pose and lanmark observation position & inverse distance
      Eigen::aligned_vector<RelLinData> rld_vec;
      linearizeHelper(rld_vec, lmdb.getObservations(), rld_error);

      // hm: the DenseAccumulator stores the H and b matrix, and provides summing (reduce) and solving 
      // hm: initialise DenseAccumulator's H and b matrix size, as it is fully determined by how many states presented in the probabilistic graph model
      BundleAdjustmentBase::LinearizeAbsReduce<DenseAccumulator<double>> lopt(
          aom);

      tbb::blocked_range<Eigen::aligned_vector<RelLinData>::iterator> range(
          rld_vec.begin(), rld_vec.end());
      // hm: iterate through all landmark observations, to add to lopt's DenseAccumulator
      tbb::parallel_reduce(range, lopt);

      double marg_prior_error = 0;
      double imu_error = 0, bg_error = 0, ba_error = 0;
      // hm: add in H and b for full states' IMU
      linearizeAbsIMU(aom, lopt.accum.getH(), lopt.accum.getB(), imu_error,
                      bg_error, ba_error, frame_states, imu_meas,
                      gyro_bias_weight, accel_bias_weight, g);
      linearizeMargPrior(marg_order, marg_H, marg_b, aom, lopt.accum.getH(),
                         lopt.accum.getB(), marg_prior_error);

      double error_total =
          rld_error + imu_error + marg_prior_error + ba_error + bg_error;

      if (config.vio_debug)
        std::cout << "[LINEARIZE] Error: " << error_total << " num points "
                  << std::endl;

      lopt.accum.setup_solver();
      Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

      bool converged = false;

      if (config.vio_use_lm) {  // Use Levenberg–Marquardt
        bool step = false;
        int max_iter = 10;

        while (!step && max_iter > 0 && !converged) {
          Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
          for (int i = 0; i < Hdiag_lambda.size(); i++)
            Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

          const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);
          double max_inc = inc.array().abs().maxCoeff();
          if (max_inc < 1e-4) converged = true;

          backup();

          // apply increment to poses
          for (auto& kv : frame_poses) {
            int idx = aom.abs_order_map.at(kv.first).first;
            kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
          }

          // apply increment to states
          for (auto& kv : frame_states) {
            int idx = aom.abs_order_map.at(kv.first).first;
            kv.second.applyInc(-inc.segment<POSE_VEL_BIAS_SIZE>(idx));
          }

          // Update points
          tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
          auto update_points_func = [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
              const auto& rld = rld_vec[i];
              updatePoints(aom, rld, inc);
            }
          };
          tbb::parallel_for(keys_range, update_points_func);

          double after_update_marg_prior_error = 0;
          double after_update_vision_error = 0, after_update_imu_error = 0,
                 after_bg_error = 0, after_ba_error = 0;

          computeError(after_update_vision_error);
          computeImuError(aom, after_update_imu_error, after_bg_error,
                          after_ba_error, frame_states, imu_meas,
                          gyro_bias_weight, accel_bias_weight, g);
          computeMargPriorError(marg_order, marg_H, marg_b,
                                after_update_marg_prior_error);

          double after_error_total =
              after_update_vision_error + after_update_imu_error +
              after_update_marg_prior_error + after_bg_error + after_ba_error;

          double f_diff = (error_total - after_error_total);

          if (f_diff < 0) {
            if (config.vio_debug)
              std::cout << "\t[REJECTED] lambda:" << lambda
                        << " f_diff: " << f_diff << " max_inc: " << max_inc
                        << " Error: " << after_error_total << std::endl;
            lambda = std::min(max_lambda, lambda_vee * lambda);
            lambda_vee *= 2;

            restore();
          } else {
            if (config.vio_debug)
              std::cout << "\t[ACCEPTED] lambda:" << lambda
                        << " f_diff: " << f_diff << " max_inc: " << max_inc
                        << " Error: " << after_error_total << std::endl;

            lambda = std::max(min_lambda, lambda / 3);
            lambda_vee = 2;

            step = true;
          }
          max_iter--;
        }

        if (config.vio_debug && converged) {
          std::cout << "[CONVERGED]" << std::endl;
        }
      } else {  // Use Gauss-Newton
        Eigen::VectorXd Hdiag_lambda = Hdiag * min_lambda;
        for (int i = 0; i < Hdiag_lambda.size(); i++)
          Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

        const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);
        double max_inc = inc.array().abs().maxCoeff();
        if (max_inc < 1e-4) converged = true;

        // apply increment to poses
        for (auto& kv : frame_poses) {
          int idx = aom.abs_order_map.at(kv.first).first;
          kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
        }

        // apply increment to states
        for (auto& kv : frame_states) {
          int idx = aom.abs_order_map.at(kv.first).first;
          kv.second.applyInc(-inc.segment<POSE_VEL_BIAS_SIZE>(idx));
        }

        // Update points
        tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
        auto update_points_func = [&](const tbb::blocked_range<size_t>& r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            const auto& rld = rld_vec[i];
            updatePoints(aom, rld, inc);
          }
        };
        tbb::parallel_for(keys_range, update_points_func);
      }

      if (config.vio_debug) {
        double after_update_marg_prior_error = 0;
        double after_update_vision_error = 0, after_update_imu_error = 0,
               after_bg_error = 0, after_ba_error = 0;

        computeError(after_update_vision_error);
        computeImuError(aom, after_update_imu_error, after_bg_error,
                        after_ba_error, frame_states, imu_meas,
                        gyro_bias_weight, accel_bias_weight, g);
        computeMargPriorError(marg_order, marg_H, marg_b,
                              after_update_marg_prior_error);

        double after_error_total =
            after_update_vision_error + after_update_imu_error +
            after_update_marg_prior_error + after_bg_error + after_ba_error;

        double error_diff = error_total - after_error_total;

        auto t2 = std::chrono::high_resolution_clock::now();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        std::cout << "iter " << iter
                  << " before_update_error: vision: " << rld_error
                  << " imu: " << imu_error << " bg_error: " << bg_error
                  << " ba_error: " << ba_error
                  << " marg_prior: " << marg_prior_error
                  << " total: " << error_total << std::endl;

        std::cout << "iter " << iter << "  after_update_error: vision: "
                  << after_update_vision_error
                  << " imu: " << after_update_imu_error
                  << " bg_error: " << after_bg_error
                  << " ba_error: " << after_ba_error
                  << " marg prior: " << after_update_marg_prior_error
                  << " total: " << after_error_total << " error_diff "
                  << error_diff << " time : " << elapsed.count()
                  << "(us),  num_states " << frame_states.size()
                  << " num_poses " << frame_poses.size() << std::endl;

        if (after_error_total > error_total) {
          std::cout << "WARN: increased error after update!!!" << std::endl;
        }
      }

      if (iter == config.vio_filter_iteration) {
        // hm: vio_outlier_threshold is passed to computeError
        // hm: minimum number of observations is set to 4
        filterOutliers(config.vio_outlier_threshold, 4);
      }

      if (converged) break;

      // std::cerr << "LT\n" << LT << std::endl;
      // std::cerr << "z_p\n" << z_p.transpose() << std::endl;
      // std::cerr << "inc\n" << inc.transpose() << std::endl;
    }
  }

  if (config.vio_debug) {
    std::cout << "==============optimize() ends==================" << std::endl;
  }
}  // namespace basalt

void KeypointVioEstimator::computeProjections(
    std::vector<Eigen::aligned_vector<Eigen::Vector4d>>& data) const {
  
  // hm: lmdb.getObservations() are organised by camera frame to which the landmarks themselves belong
  for (const auto& kv : lmdb.getObservations()) {
    // hm: tcid_h is the host of the landmark
    const TimeCamId& tcid_h = kv.first;
    // hm: obs_kv is the enumeration of each time-camera frame, and its corresponding observations
    for (const auto& obs_kv : kv.second) {
      // hm: tcid_t is the observing frame
      const TimeCamId& tcid_t = obs_kv.first;
      // hm: only process observations of the current frame
      if (tcid_t.frame_id != last_state_t_ns) continue;

      // hm: when landmark is observed in the non-host frame
      if (tcid_h != tcid_t) {
        PoseStateWithLin state_h = getPoseStateWithLin(tcid_h.frame_id);
        PoseStateWithLin state_t = getPoseStateWithLin(tcid_t.frame_id);

        Sophus::SE3d T_t_h_sophus =
            computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                           state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);

        Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

        // hm: T_t_h describe the camera poses change between the landmark's host frame and the observing target frame
        FrameRelLinData rld;

        std::visit(
            [&](const auto& cam) {
              for (size_t i = 0; i < obs_kv.second.size(); i++) {
                // hm: the image coordinates of a observation
                const KeypointObservation& kpt_obs = obs_kv.second[i];
                // hm: the position info of that keypoint corresponding to the observation
                const KeypointPosition& kpt_pos =
                    lmdb.getLandmark(kpt_obs.kpt_id);

                Eigen::Vector2d res;
                Eigen::Vector4d proj;

                linearizePoint(kpt_obs, kpt_pos, T_t_h, cam, res, nullptr,
                               nullptr, &proj);
                // hm: the third one in proj[] is inverse distance
                // hm: the forth one in proj[] is the KEYPOINT ID
                proj[3] = kpt_obs.kpt_id;
                data[tcid_t.cam_id].emplace_back(proj);
              }
            },
            calib.intrinsics[tcid_t.cam_id].variant);

      } else {
        // target and host are the same
        // residual does not depend on the pose
        // it just depends on the point

        std::visit(
            [&](const auto& cam) {
              for (size_t i = 0; i < obs_kv.second.size(); i++) {
                const KeypointObservation& kpt_obs = obs_kv.second[i];
                const KeypointPosition& kpt_pos =
                    lmdb.getLandmark(kpt_obs.kpt_id);

                Eigen::Vector2d res;
                Eigen::Vector4d proj;

                linearizePoint(kpt_obs, kpt_pos, Eigen::Matrix4d::Identity(),
                               cam, res, nullptr, nullptr, &proj);

                proj[3] = kpt_obs.kpt_id;
                data[tcid_t.cam_id].emplace_back(proj);
              }
            },
            calib.intrinsics[tcid_t.cam_id].variant);
      }
    }
  }
}

}  // namespace basalt

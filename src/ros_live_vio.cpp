#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <thread>
#include <stdlib.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tbb.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/vis_utils.h>

#include <ros/ros.h>
#include <basalt/imu/imu_types.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <basalt/io/stereo_processor.h>

// GUI functions
void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene();
void load_data(const std::string& calib_path);
void draw_plots();

// Pangolin variables
constexpr int UI_WIDTH = 200;

using Button = pangolin::Var<std::function<void(void)>>;

pangolin::DataLog imu_data_log, vio_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

pangolin::Var<bool> show_gt("ui.show_gt", true, false, true);

pangolin::Var<bool> follow("ui.follow", true, false, true);

// Visualization variables
basalt::VioVisualizationData::Ptr curr_vis_data;

tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState::Ptr> out_state_queue;
tbb::concurrent_bounded_queue<basalt::ImuData::Ptr>* imu_data_queue = nullptr;

std::vector<int64_t> vio_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

std::mutex m;
bool step_by_step = false;
int64_t last_t_ns = -1;
int64_t first_t_ns = -1;


// VIO variables
basalt::Calibration<double> calib;

basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;
basalt::VioEstimatorBase::Ptr vio;
basalt::OpticalFlowInput::Ptr last_img_data;

void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg){

  static int64_t pre_imu_seq = 0;
  static int64_t pre_ts = 0;
  // std::cout<<" got imu msgs"<<std::endl;
  if(!pre_imu_seq) {
    pre_imu_seq = imu_msg->header.seq;
    pre_ts = imu_msg->header.stamp.toNSec();
    return;
  }
  // std::cout<<"pre_imu_seq: "<<pre_imu_seq<<", cur_imu_seq: "<<imu_msg->header.seq<<std::endl;
  if(imu_msg->header.seq != pre_imu_seq + 1){
    std::cout << "IMU packet loss, sequence number not continuous, now" << imu_msg->header.seq << " and previous " << pre_imu_seq << std::endl;
    throw std::runtime_error("abort because of bad IMU stream");
  }
  pre_imu_seq = imu_msg->header.seq;

  basalt::ImuData::Ptr data(new basalt::ImuData);
  data->t_ns = imu_msg->header.stamp.toNSec();

  // 1 second jump
  if (pre_ts >= data->t_ns || data->t_ns - pre_ts >= 1000e6 ){
    std::cout << "IMU time jump detected, aborting()" << std::endl;
    std::cout << "pre_ts = " << double(pre_ts) / 1e9 << ", now_ts = " << double(data->t_ns) / 1e9  << std::endl;
    abort();
  }
  pre_ts = data->t_ns;

  data->accel = Eigen::Vector3d(
          imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y,
          imu_msg->linear_acceleration.z);
  data->gyro = Eigen::Vector3d(
          imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
          imu_msg->angular_velocity.z);
  if (imu_data_queue) {
    if(imu_data_queue->try_push(data)){
      // if(vio_config.vio_debug)
      //   std::cout<< "got imu msg at time "<< imu_msg->header.stamp <<std::endl;
    }
    else{
      std::cout<<"imu data buffer is full: "<<imu_data_queue->size()<<std::endl;
      // abort();
    }
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "vio_ros");
	ros::NodeHandle nh;
  ros::NodeHandle local_nh("~");
  std::string cam_calib_path;
  std::string config_path;
  bool terminate = false;
  bool show_gui = true;
  bool print_queue = false;
  int num_threads = 0;
  bool use_imu = true;
  local_nh.param<std::string>("calib_file", cam_calib_path, "basalt_ws/src/basalt/data/zed_calib.json");
  local_nh.param<std::string>("config_path", config_path, "basalt_ws/src/basalt/data/zed_config.json");
  local_nh.param("show_gui", show_gui, true);
  local_nh.param("print_queue", print_queue, false);
  local_nh.param("terminate", terminate, false);
  local_nh.param("use_imu", use_imu, true);

  if (!config_path.empty()) {
    vio_config.load(config_path);
  } else {
    vio_config.optical_flow_skip_frames = 2;
  }

  StereoProcessor::Parameters stereoParam;
  stereoParam.queue_size = 3;
  stereoParam.left_topic = "/zed/left/image_raw_color";
  stereoParam.right_topic = "/zed/right/image_raw_color";
  stereoParam.left_info_topic = "/zed/left/camera_info_raw";
  stereoParam.right_info_topic = "/zed/right/camera_info_raw";
  StereoProcessor stereo_sub(vio_config, stereoParam);
  last_img_data = stereo_sub.last_img_data;
  ros::Subscriber Imusub = nh.subscribe("/mavros/imu/data/sys_id_9", 200, imuCallback); // 2 seconds of buffering
  ros::AsyncSpinner spinner(4);
  spinner.start();

  if (num_threads > 0) {
    tbb::task_scheduler_init init(num_threads);
  }

  //  load calibration
  load_data(cam_calib_path);

  std::cout<<"calib.T_i_c: " << calib.T_i_c[0].translation().x() << ","
              << calib.T_i_c[0].translation().y() << ","
              << calib.T_i_c[0].translation().z() << ","
              << calib.T_i_c[0].unit_quaternion().w() << ","
              << calib.T_i_c[0].unit_quaternion().x() << ","
              << calib.T_i_c[0].unit_quaternion().y() << ","
              << calib.T_i_c[0].unit_quaternion().z() << std::endl<<std::endl;


  opt_flow_ptr = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);
  stereo_sub.image_data_queue = &opt_flow_ptr->input_queue;

  vio = basalt::VioEstimatorFactory::getVioEstimator(
      vio_config, calib, basalt::constants::g, use_imu);

  vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  imu_data_queue = &vio->imu_data_queue;

  opt_flow_ptr->output_queue = &vio->vision_data_queue;
  if (show_gui) vio->out_vis_queue = &out_vis_queue;
  vio->out_state_queue = &out_state_queue;

  vio_data_log.Clear();

  std::shared_ptr<std::thread> t3;

  if (show_gui)
    t3.reset(new std::thread([&]() {
      while (true) {
        out_vis_queue.pop(curr_vis_data);

        if (!curr_vis_data.get()) break;
      }

      std::cout << "Finished t3" << std::endl;
    }));

  std::thread t4([&]() {
    basalt::PoseVelBiasState::Ptr data;
    ros::Publisher pose_cov_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/basalt/pose_nwu", 10);
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/basalt/pose_cov_nwu", 10);
    ros::Publisher pose_map_pub = nh.advertise<geometry_msgs::PoseStamped>("/basalt/pose_enu", 10);
    ros::Publisher pose_cov_map_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/basalt/pose_cov_enu", 10);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/basalt/odom_nwu", 10);
    ros::Publisher odom_ned_pub = nh.advertise<nav_msgs::Odometry>("/basalt/odom_ned", 10);

    try{

        while (true) {
        out_state_queue.pop(data);

        if (!data.get()) break;

        int64_t t_ns = data->t_ns;

        Sophus::SE3d T_w_i = data->T_w_i;
        Eigen::Vector3d vel_w_i = data->vel_w_i;
        Eigen::Vector3d bg = data->bias_gyro;
        Eigen::Vector3d ba = data->bias_accel;

        vio_t_ns.emplace_back(data->t_ns);

        if (last_t_ns > 0){

          // hm: abort if speed greater than 15 m/s, probably a failure
          if (vel_w_i.norm() > 15 )
          {
            std::cout << "detect speed too fast > 15 m/s : " << vel_w_i.norm() << std::endl;
            abort();
          }

          // hm: detect big change in estimated position accross frames, > 4m accross frame
          if ( (vio_t_w_i.back() - T_w_i.translation()).norm() > 3){
            std::cout << "detect translation change > than 3:  " << (vio_t_w_i.back() - T_w_i.translation()).norm() << std::endl;
            abort();
          }
        }else{
          first_t_ns = t_ns;
        }
        last_t_ns = t_ns;
        
        vio_t_w_i.emplace_back(T_w_i.translation());

        // the w_i here is referring to vision world (for now, it is the same as IMU frame, which is FLU / NWU )
        // we want to follow ROS convention on the map coordinate, which is NEU

        Sophus::Matrix3d R_m_w, R_ned_nwu;

        // change of coordinates from NWU to ENU
        R_m_w <<  0,-1,0,
                  1,0,0,
                  0,0,1; 

        // change of coordinates from NWU to NED
        R_ned_nwu <<  1,0,0,
                    0,-1,0,
                    0,0,-1;


        // reference: https://dev.px4.io/master/en/ros/external_position_estimation.html#ros_reference_frames
        // T_w_i: i is in NWU, and w(basalt local frame) is in NWU
        // T_m_i: i is in NUW, and m(ROS map) is in ENU

        // for ROS
        Sophus::SE3d T_m_i;
        T_m_i.translation() = R_m_w * T_w_i.translation();
        T_m_i.setRotationMatrix(R_m_w * T_w_i.rotationMatrix() * R_m_w.inverse());

        // for MAVLink Odom message
        Sophus::SE3d T_ned_frd;
        T_ned_frd.translation() = R_ned_nwu * T_w_i.translation();
        T_ned_frd.setRotationMatrix(R_ned_nwu * T_w_i.rotationMatrix() * R_ned_nwu.inverse());

        // vel_w_i is in NWU
        Eigen::Vector3d vel_body_ned =  R_ned_nwu * T_w_i.rotationMatrix().inverse() * vel_w_i;


        geometry_msgs::Pose pose, pose_enu, pose_ned;
        geometry_msgs::Twist twist, twist_ned;
        nav_msgs::Odometry odom, odom_ned;

        // basalt frame
        {
          pose.position.x =  T_w_i.translation()[0];
          pose.position.y =  T_w_i.translation()[1];
          pose.position.z =  T_w_i.translation()[2];
          pose.orientation.w = T_w_i.unit_quaternion().w();
          pose.orientation.x = T_w_i.unit_quaternion().x();
          pose.orientation.y = T_w_i.unit_quaternion().y();
          pose.orientation.z = T_w_i.unit_quaternion().z();

          twist.linear.x = vel_w_i[0];
          twist.linear.y = vel_w_i[1];
          twist.linear.z = vel_w_i[2];
        }

        // ROS ENU frame
        {
          pose_enu.position.x =  T_m_i.translation()[0];
          pose_enu.position.y =  T_m_i.translation()[1];
          pose_enu.position.z =  T_m_i.translation()[2];
          pose_enu.orientation.w = T_m_i.unit_quaternion().w();
          pose_enu.orientation.x = T_m_i.unit_quaternion().x();
          pose_enu.orientation.y = T_m_i.unit_quaternion().y();
          pose_enu.orientation.z = T_m_i.unit_quaternion().z();
        }

        // PX4 NED frame
        {
          pose_ned.position.x = T_ned_frd.translation()[0];
          pose_ned.position.y = T_ned_frd.translation()[1];
          pose_ned.position.z = T_ned_frd.translation()[2];
          pose_ned.orientation.w = T_ned_frd.unit_quaternion().w();
          pose_ned.orientation.x = T_ned_frd.unit_quaternion().x();
          pose_ned.orientation.y = T_ned_frd.unit_quaternion().y();
          pose_ned.orientation.z = T_ned_frd.unit_quaternion().z();

          twist_ned.linear.x = vel_body_ned[0];
          twist_ned.linear.y = vel_body_ned[1];
          twist_ned.linear.z = vel_body_ned[2];
        }

        // pose in local world frame
        {
          geometry_msgs::PoseStamped poseMsg;
          poseMsg.header.stamp.fromNSec(t_ns);
          poseMsg.header.frame_id = "odom";
          poseMsg.pose = pose;
          pose_pub.publish(poseMsg);
        }

        // pose with covariance in local world frame
        {
          geometry_msgs::PoseWithCovarianceStamped poseMsg;
          poseMsg.header.stamp.fromNSec(t_ns);
          poseMsg.header.frame_id = "odom";
          poseMsg.pose.pose =  pose;

          odom.header = poseMsg.header;
          odom.child_frame_id = "base_link";
          odom.pose.pose = pose;
          odom.twist.twist = twist;

          pose_cov_pub.publish(poseMsg);
          odom_pub.publish(odom);
        }

        // pose in ROS enu world frame
        {
          geometry_msgs::PoseStamped poseMsg;
          poseMsg.header.stamp.fromNSec(t_ns);
          poseMsg.header.frame_id = "map";
          poseMsg.pose = pose_enu;
          pose_map_pub.publish(poseMsg);
        }

        // pose with covariance in ROS enu world frame
        {
          geometry_msgs::PoseWithCovarianceStamped poseMsg;
          poseMsg.header.stamp.fromNSec(t_ns);
          poseMsg.header.frame_id = "map";
          poseMsg.pose.pose = pose_enu;
          pose_cov_map_pub.publish(poseMsg);
          
        }

        {
          odom_ned.header.stamp.fromNSec(t_ns);
          odom_ned.header.frame_id = "odom_ned";
          odom_ned.child_frame_id = "base_link_frd";
          odom_ned.pose.pose = pose_ned;
          odom_ned.twist.twist = twist_ned;
          odom_ned_pub.publish(odom_ned);
        }
          
        

        if (show_gui) {
          std::vector<float> vals;
          vals.push_back((t_ns - first_t_ns) * 1e-9);

          for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
          for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
          for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
          for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

          vio_data_log.Log(vals);
        }
      }
      
    }catch(const std::exception& e){
      throw std::runtime_error("visualisation thread runtime error");
    }

    

    std::cout << "Finished t4" << std::endl;
  });

  std::shared_ptr<std::thread> t5;

  if (print_queue) {
    t5.reset(new std::thread([&]() {
      while (!terminate) {
        std::cout << "opt_flow_ptr->input_queue "
                  << opt_flow_ptr->input_queue.size()
                  << " opt_flow_ptr->output_queue "
                  << opt_flow_ptr->output_queue->size() << " out_state_queue "
                  << out_state_queue.size() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }));
  }

  if (show_gui) {
    pangolin::CreateWindowAndBind("ROS Live Vio", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.4)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    plotter =
        new pangolin::Plotter(&imu_data_log, 0.0, 100, -3.0, 3.0, 0.01f, 0.01f);
    plot_display.AddDisplay(*plotter);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    Eigen::Vector3d cam_p(2, -8, -8);
    cam_p = vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;
    cam_p[2] = 4;

    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0,
                                  pangolin::AxisZ));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.4, 1.0, 0.4, 1.0)
            .SetHandler(new pangolin::Handler3D(camera));

    while (ros::ok()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (follow) {
        if (curr_vis_data.get()) {
          auto T_w_i = curr_vis_data->states.back();
          T_w_i.so3() = Sophus::SO3d();

          camera.Follow(T_w_i.matrix());
        }
      }

      display3D.Activate(camera);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      draw_scene();

      img_view_display.Activate();

      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_SHORT;
        fmt.scalable_internal_format = GL_LUMINANCE16;

        if (curr_vis_data.get() && curr_vis_data->opt_flow_res.get() &&
            curr_vis_data->opt_flow_res->input_images.get()) {
          auto& img_data = curr_vis_data->opt_flow_res->input_images->img_data;

          for (size_t cam_id = 0; cam_id < 2;
               cam_id++) {
            if (img_data[cam_id].img.get())
              img_view[cam_id]->SetImage(
                  img_data[cam_id].img->ptr, img_data[cam_id].img->w,
                  img_data[cam_id].img->h, img_data[cam_id].img->pitch, fmt);
          }
        }

        draw_plots();
      }

      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
          show_est_ba.GuiChanged() || show_est_bg.GuiChanged()) {
        draw_plots();
      }

      pangolin::FinishFrame();
    }
  }else
    ros::waitForShutdown();

  terminate = true;
  std::cout<<"terminate!!!"<<std::endl;

  if (stereo_sub.image_data_queue) stereo_sub.image_data_queue->push(nullptr);
  if (imu_data_queue) imu_data_queue->push(nullptr);

  if (t3.get()) t3->join();
  t4.join();
  if (t5.get()) t5->join();

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (curr_vis_data.get() && cam_id < curr_vis_data->projections.size()) {
      const auto& points = curr_vis_data->projections[cam_id];
      const auto& optical_flow_obs = curr_vis_data->opt_flow_res->observations[cam_id];

      double min_id = 1, max_id = 1;

      if (!points.empty()) {
        min_id = points[0][2];
        max_id = points[0][2];

        // hm: for each point, first 2 number is the coordinate, 3rd number is the inverse distance
        // hm: the last number not used?
        for (const auto& points2 : curr_vis_data->projections)
          for (const auto& p : points2) {
            min_id = std::min(min_id, p[2]);
            max_id = std::max(max_id, p[2]);
          }

        //hm: set the coloring to be constant
        // min_id = 0.002; // blue color
        // max_id = 1; // red color

        const double blue_id = 0.002; // blue color
        const double red_id = 0.5; // red color  

        for (const auto& c : points) {
          const float radius = 6.5; 

          float r, g, b;
          double scale = c[2] - blue_id;
          if (scale < 0.0)
            scale = 0;
          else if (c[2] > red_id)
            scale = red_id - blue_id;
          getcolor(scale , red_id - blue_id, b, g, r);
          glColor3f(r, g, b);

          pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

          // hm: above plot the observation after optimisation, now we want to project the original observation too
          // hm: the fourth element of the visualisation data, shows the id keypoint
          const uint32_t kpt_id = int(c[3]);

          if (show_ids)
            pangolin::GlFont::I().Text("%u", kpt_id).Draw(c[0], c[1]);

          // hm: in rare cases, if the landmark is no longer observed, skip
          if(!optical_flow_obs.count(kpt_id))
            continue;

          auto vec = optical_flow_obs.at(kpt_id).translation().cast<double>();

          pangolin::glDrawCircle(vec, 1.0);

          if(vio_config.vio_debug){
            if(!calib.intrinsics[cam_id].inBound(c.head(2))){
              std::cout << c.transpose() << " optimised point is out of bound at cam " << cam_id << std::endl;
              // abort();
            }
            if(!calib.intrinsics[cam_id].inBound(vec)){
              std::cout << vec << " flow obs is out of bound at cam " << cam_id << std::endl;
              abort();
            }
          }
          pangolin::glDrawLine(c[0], c[1],vec[0], vec[1]);
          
        }
      }

      glColor3f(1.0, 0.8, 0.0);
      pangolin::GlFont::I()
          .Text("Tracked %d points, mixd = %.2lf maxd = %.2lf", points.size(), 1/max_id, 1/min_id)
          .Draw(5, 40);
    }
  }
}

void draw_scene() {
  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(cam_color);
  Eigen::aligned_vector<Eigen::Vector3d> sub_gt(vio_t_w_i.begin(),
                                                vio_t_w_i.end());
  pangolin::glDrawLineStrip(sub_gt);

  if (curr_vis_data.get()) {
    for (const auto& p : curr_vis_data->states)
      for (const auto& t_i_c : calib.T_i_c)
        render_camera((p * t_i_c).matrix(), 2.0f, state_color, 0.1f);

    for (const auto& p : curr_vis_data->frames)
      for (const auto& t_i_c : calib.T_i_c)
        render_camera((p * t_i_c).matrix(), 2.0f, pose_color, 0.1f);

    for (const auto& t_i_c : calib.T_i_c)
      render_camera((curr_vis_data->states.back() * t_i_c).matrix(), 2.0f,
                    cam_color, 0.1f);

    glColor3ubv(pose_color);
    pangolin::glDrawPoints(curr_vis_data->points);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}

void draw_plots() {
  plotter->ClearSeries();
  plotter->ClearMarkers();

  if (show_est_pos) {
    plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "position x", &vio_data_log);
    plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "position y", &vio_data_log);
    plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "position z", &vio_data_log);
  }

  if (show_est_vel) {
    plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "velocity x", &vio_data_log);
    plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "velocity y", &vio_data_log);
    plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "velocity z", &vio_data_log);
  }

  if (show_est_bg) {
    plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "gyro bias x", &vio_data_log);
    plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "gyro bias y", &vio_data_log);
    plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
  }

  if (show_est_ba) {
    plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "accel bias x", &vio_data_log);
    plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "accel bias y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "accel bias z", &vio_data_log);
  }

  if (last_img_data.get()) {
    double t = last_img_data->t_ns * 1e-9;
    plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                       pangolin::Colour::White());
  }
}

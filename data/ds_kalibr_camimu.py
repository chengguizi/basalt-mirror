#!/usr/bin/env python3

import yaml
import argparse

import numpy as np
from string import Template
from scipy.spatial.transform import Rotation as R
import sophus as sp

### Obtain the Path
parser = argparse.ArgumentParser(description='Convert Kalibr Calibration to Basalt-like parameters')
parser.add_argument('yaml', type=str, help='Kalibr Yaml file path')
parser.add_argument('output_name', type=str, help='Output name of the json file')
args = parser.parse_args()
print(args.yaml)
#tis param
                # "px": 0.03,
                # "py": 0,
                # "pz": 0,
                # "qx": 0,
                # "qy": 0,
                # "qz": 1,
                # "qw": 0

calib_template = Template('''{
    "value0": {
        "T_imu_cam": [
            {
                "px": $px0,
                "py": $py0,
                "pz": $pz0,
                "qx": $qx0,
                "qy": $qy0,
                "qz": $qz0,
                "qw": $qw0
            },
            {
                "px": $px1,
                "py": $py1,
                "pz": $pz1,
                "qx": $qx1,
                "qy": $qy1,
                "qz": $qz1,
                "qw": $qw1
            }
        ],
        "intrinsics": [
            {
                "camera_type": "ds",
                "intrinsics": {
                    "fx": $fx0,
                    "fy": $fy0,
                    "cx": $cx0,
                    "cy": $cy0,
                    "xi": $xi0,
                    "alpha": $alpha0
                }
            },
            {
                "camera_type": "ds",
                "intrinsics": {
                    "fx": $fx1,
                    "fy": $fy1,
                    "cx": $cx1,
                    "cy": $cy1,
                    "xi": $xi1,
                    "alpha": $alpha1
                }
            }
        ],
        "resolution": [
            [
                $rx,
                $ry
            ],
            [
                $rx,
                $ry
            ]
        ],
        "vignette": [],
        "calib_accel_bias": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "calib_gyro_bias": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "imu_update_rate": $imu_rate,
        "accel_noise_std": [0.016, 0.016, 0.016],
        "gyro_noise_std": [0.000282, 0.000282, 0.000282],
        "accel_bias_std": [0.001, 0.001, 0.001],
        "gyro_bias_std": [0.0001, 0.0001, 0.0001],
        "cam_time_offset_ns": $toffset
    }
}
''')

stream = open(args.yaml, 'r')
# stream = open("/media/nvidia/SD/catkin_ws/src/basalt-mirror/data/tis_23/camchain-imucam-2020-08-08-16-00-21.yaml", 'r')


f = yaml.load(stream)
stream.close()

# obtain imu to camx transformation
T_c0_imu = sp.SE3(f['cam0']['T_cam_imu'])
T_c1_imu = sp.SE3(f['cam1']['T_cam_imu'])

print('imu in camera 0 transformation:')
print(T_c0_imu)

print('imu in camera 1 transformation:')
print(T_c1_imu)

print('camera 0 in imu transformation')
# assume IMU is in NWU frame and is mounting facing forward
# assume the two cameras are mounted forward too. frame right-down-forward
# R_imu_c0 = sp.SO3([ [ 0, 0, 1],
#                     [-1, 0, 0],
#                     [ 0,-1, 0]])
# t_imu_c0 = [0.1, 0.1 , 0]
T_imu_c0 = T_c0_imu.inverse()
print(T_imu_c0)

t_imu_c0 = T_imu_c0.translation()
q_imu_c0 = R.from_matrix(T_imu_c0.rotationMatrix()).as_quat()


print('camera 1 in imu transformation')
T_imu_c1 = T_c1_imu.inverse()
print(T_imu_c1)

t_imu_c1 = T_imu_c1.translation()
q_imu_c1 = R.from_matrix(T_imu_c1.rotationMatrix()).as_quat()


distort_0 = f['cam0']['distortion_coeffs']
distort_1 = f['cam1']['distortion_coeffs']

intrinsics_0 = f['cam0']['intrinsics']
intrinsics_1 = f['cam1']['intrinsics']

resolution_0 = f['cam0']['resolution']
resolution_1 = f['cam1']['resolution']

# timu = tcam + toffset in seconds
toffset = round(f['cam0']['timeshift_cam_imu'] * 1e9)

# transformations are all respect to imu frame
values = {'px0':  t_imu_c0[0] , 'py0':  t_imu_c0[1]  ,'pz0':  t_imu_c0[2]  ,
            'px1':  t_imu_c1[0] , 'py1':  t_imu_c1[1] , 'pz1':  t_imu_c1[2]  ,
            'qx0':  q_imu_c0[0] , 'qy0':  q_imu_c0[1] , 'qz0':  q_imu_c0[2] , 'qw0':  q_imu_c0[3] ,
            'qx1':  q_imu_c1[0] , 'qy1':  q_imu_c1[1] , 'qz1':  q_imu_c1[2] , 'qw1':  q_imu_c1[3] ,
            'fx0': intrinsics_0[2], 'fy0': intrinsics_0[3], 'cx0': intrinsics_0[4], 'cy0': intrinsics_0[5], 'xi0': intrinsics_0[0],'alpha0': intrinsics_0[1],
            'fx1': intrinsics_1[2], 'fy1': intrinsics_1[3], 'cx1': intrinsics_1[4], 'cy1': intrinsics_1[5], 'xi1': intrinsics_1[0],'alpha1': intrinsics_1[1],
            'rx': resolution_0[0], 'ry': resolution_0[1],
            'imu_rate' : 180.0, 'toffset':  toffset}


calib = calib_template.substitute(values)
print(calib)

with open('./'+ args.output_name + '.json', 'w') as stream2:
    stream2.write(calib)

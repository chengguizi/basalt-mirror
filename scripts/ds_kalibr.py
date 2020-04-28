#!/usr/bin/env python

import yaml
import argparse

import numpy as np
from string import Template
from scipy.spatial.transform import Rotation as R

### Obtain the Path
# parser = argparse.ArgumentParser(description='Convert Kalibr Calibration to Basalt-like parmeters')
# parser.add_argument('yaml', type=str, help='Kalibr Yaml file path')
# args = parser.parse_args()
# print(args.yaml)
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
                "pz": 0,
                "qx": $qx0,
                "qy": $qy0,
                "qz": $qz0,
                "qw": $qw0
            },
            {
                "px": $px1,
                "py": $py1,
                "pz": 0,
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
        "cam_time_offset_ns": 0
    }
}
''')

# stream = open(args.yaml, 'r')
stream = open("/home/yu/Documents/data/tis_kalibr/camchain-imucam-2020-03-13-10-30-44_tis.yaml", 'r')


f = yaml.load(stream)
stream.close()

# Extract cam1 to imu from cam0 to cam1
# T_c1_c0 = np.matrix(f['cam1']['T_cn_cnm1'])
# r_c0_c1 = np.linalg.inv(T_c1_c0[0:3,0:3])
# R_c0_c1 = R.from_matrix(r_c0_c1)
# r_i_c0 = np.array([[-1, 0, 0], [0, -1, 0],[0, 0, 1]])
# R_i_c0 = R.from_matrix(r_i_c0)
# # print(R_i_c0.as_quat())
# R_i_c1 = (R_i_c0 * R_c0_c1).as_quat()
# t_c0_c1 = -r_c0_c1.dot(T_c1_c0[0:3,3])
# T_i_c1 = r_i_c0.dot(t_c0_c1)
# 'px1': T_i_c1.item(0) + 0.03, 'py1': T_i_c1.item(1), 'pz1': T_i_c1.item(2),
            # 'qx1': R_i_c1[0] , 'qy1': R_i_c1[1] , 'qz1': R_i_c1[2] , 'qw1': R_i_c1[3] 

# inverse version
T_cam_imu_0 = np.matrix(f['cam0']['T_cam_imu'])
R_inv_0 = np.linalg.inv(T_cam_imu_0[0:3,0:3])
# print(R_inv_0.dot(T_cam_imu_0[0:3,0:3]))
r = R.from_matrix(R_inv_0)
# r_inv = r.inv()
# print(r.as_matrix())
# print(r_inv.as_matrix()- T_cam_imu_0[0:3,0:3])
q_0 = r.as_quat()
# print(q_0)
t_inv_0 = -R_inv_0.dot(T_cam_imu_0[0:3, 3])

T_cam_imu_1 = np.matrix(f['cam1']['T_cam_imu'])
R_inv_1 = np.linalg.inv(T_cam_imu_1[0:3,0:3])
r = R.from_matrix(R_inv_1)
q_1 = r.as_quat()
# print(q_1)
t_inv_1 = -R_inv_1.dot(T_cam_imu_1[0:3, 3])


distort_0 = f['cam0']['distortion_coeffs']
distort_1 = f['cam1']['distortion_coeffs']

intrinsics_0 = f['cam0']['intrinsics']
intrinsics_1 = f['cam1']['intrinsics']

resolution_0 = f['cam0']['resolution']
resolution_1 = f['cam1']['resolution']


values = {'px0':  t_inv_0.item(0) , 'py0':  t_inv_0.item(1)  ,
            'px1':  t_inv_1.item(0) , 'py1':  t_inv_1.item(1)  ,
            'qx0':  q_0[0] , 'qy0':  q_0[1] , 'qz0':  q_0[2] , 'qw0':  q_0[3] ,
            'qx1':  q_1[0] , 'qy1':  q_1[1] , 'qz1':  q_1[2] , 'qw1':  q_1[3] ,
            'fx0': intrinsics_0[2], 'fy0': intrinsics_0[3], 'cx0': intrinsics_0[4], 'cy0': intrinsics_0[5], 'xi0': intrinsics_0[0],'alpha0': intrinsics_0[1], 
            'fx1': intrinsics_1[2], 'fy1': intrinsics_1[3], 'cx1': intrinsics_1[4], 'cy1': intrinsics_1[5], 'xi1': intrinsics_1[0],'alpha1': intrinsics_1[1], 
            'rx': resolution_0[0], 'ry': resolution_0[1],
            'imu_rate' : 100.0}


calib = calib_template.substitute(values)
print(calib)

with open('/home/yu/basalt_ws/src/basalt/data/tis_calib.json', 'w') as stream2:
    stream2.write(calib)
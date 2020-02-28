#!/usr/bin/env python

import yaml
import argparse

import numpy as np
from string import Template
from scipy.spatial.transform import Rotation as R

### Obtain the Path
parser = argparse.ArgumentParser(description='Convert Kalibr Calibration to Basalt-like parmeters')
parser.add_argument('yaml', type=str, help='Kalibr Yaml file path')
args = parser.parse_args()
# print(args.yaml)
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
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": $fx0,
                    "fy": $fy0,
                    "cx": $cx0,
                    "cy": $cy0
                }
            },
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": $fx1,
                    "fy": $fy1,
                    "cx": $cx1,
                    "cy": $cy1
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
        "accel_noise_std": [0.0, 0.0, 0.0],
        "gyro_noise_std": [0.0, 0.0, 0.0],
        "accel_bias_std": [0.0, 0.0, 0.0],
        "gyro_bias_std": [0.0, 0.0, 0.0],
        "cam_time_offset_ns": 0
    }
}
''')

stream = open(args.yaml, 'r')
f = yaml.load(stream)
stream.close()

T_cam_imu_0 = np.matrix(f['cam0']['T_cam_imu'])
R_inv_0 = np.linalg.inv(T_cam_imu_0[0:3,0:3])
# print(R_inv_0.dot(T_cam_imu_0[0:3,0:3]))
r = R.from_matrix(R_inv_0)
r_inv = r.inv()
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

timeshift_cam_imu_0 = f['cam0']['timeshift_cam_imu']
timeshift_cam_imu_1 = f['cam1']['timeshift_cam_imu']

values = {'px0': round(t_inv_0.item(0),3), 'py0': round(t_inv_0.item(1),3), 'pz0': round(t_inv_0.item(2),3),
            'px1': round(t_inv_1.item(0),3), 'py1': round(t_inv_1.item(1),3), 'pz1': round(t_inv_1.item(2),3),
            'qx0': round(q_0[0],3), 'qy0': round(q_0[1],3), 'qz0': round(q_0[2],3), 'qw0': round(q_0[3],3),
            'qx1': round(q_1[0],3), 'qy1': round(q_1[1],3), 'qz1': round(q_1[2],3), 'qw1': round(q_1[3],3),
            'fx0': round(intrinsics_0[0],3), 'fy0': round(intrinsics_0[1],3), 'cx0': round(intrinsics_0[2],3), 'cy0': round(intrinsics_0[3],3), 
            'fx1': round(intrinsics_1[0],3), 'fy1': round(intrinsics_1[1],3), 'cx1': round(intrinsics_1[2],3), 'cy1': round(intrinsics_1[3],3), 
            'rx': round(resolution_0[0],3), 'ry': round(resolution_0[1],3),
            'imu_rate' : 100.0}


calib = calib_template.substitute(values)
print(calib)

with open('/home/yu/basalt_ws/src/basalt/data/zed_calib.json', 'w') as stream2:
    stream2.write(calib)
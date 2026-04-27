import os
import sys
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from mapanything.datasets.base.base_dataset import BaseDataset
import time 
import pdb
import open3d as o3d
import copy
from datetime import datetime
import glob

class LiveRealSenseDataset(BaseDataset):
    def __init__(self,                 
                 base_client, 
                 gripper,
                 router,
                 base_dir="infer",
                 eye_in_hand_serial="043422251523",
                 hand_eye_transform_path="infer/hand_eye_transform.npy",
                 start_pose_path="infer/start.json",
                 split="test", 
                 data_norm_type="dinov2", 
                 transform="imgnorm",     
                 resolution=(518, 518),
                 num_views=4, 
                 **kwargs):
        
        super().__init__(
            split=split, data_norm_type=data_norm_type, 
            transform=transform, num_views=num_views,  
            resolution=resolution, **kwargs
        )
        infer_dir = os.path.abspath("infer") 
        gripper.open()
        
        if infer_dir not in sys.path:
            sys.path.insert(0, infer_dir)
            
        from robotic_grasping_utilities import RobotController, parse_joints_from_json      

        self.base_client = base_client
        self.eye_in_hand_serial = eye_in_hand_serial
        self.T_ee_to_eih = np.load(hand_eye_transform_path)

        print(f"Moving robot to start position from {start_pose_path}...")
        if os.path.exists(start_pose_path):
            try:
                robot_controller = RobotController(router)
                start_config = parse_joints_from_json(start_pose_path)
                
                if robot_controller.execute_trajectory([start_config[0]]):
                    print("Robot successfully reached start position.")
                    time.sleep(1.0) # Allow physical vibrations to settle before camera warmup
                else:
                    print("Warning: Robot failed to reach start position.")
            except Exception as e:
                print(f"Error moving to start position: {e}")
        else:
            print(f"Error: Start position file {start_pose_path} not found. Skipping movement.")
        # -----------------------------------------

        self.ctx = rs.context()
        self.camera_configs = []
        
        serials = ['043422251523']
        # serials = ['043422251523', '043422251106', '043422252545']

        for serial in serials:
            intrinsics = np.load(os.path.join(base_dir, f"camera_{serial}_intrinsics.npz"))
            
            static_pose = None
            if serial != eye_in_hand_serial:
                static_pose = np.load(os.path.join(base_dir, f"ext_cam_{serial}_to_base.npy"))

            print(f"Starting stream for {serial}...")
            pipe = rs.pipeline(self.ctx)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 5)
            
            profile = pipe.start(config)

            dev = profile.get_device()
            for sensor in dev.query_sensors():
                
                if sensor.is_color_sensor():
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    sensor.set_option(rs.option.enable_auto_white_balance, 0)
                    sensor.set_option(rs.option.exposure, 200) 
                    sensor.set_option(rs.option.white_balance, 4000)
                # --------------------------------------------------

            align = rs.align(rs.stream.color)
            
            self.camera_configs.append({
                "serial": serial,
                "K": intrinsics["camera_matrix"],
                "static_pose": static_pose,
                "pipe": pipe,
                "align": align,
                "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale()
            })

        print("Finalising sensor settings for all cameras...")
        for _ in range(30):
            for cam_data in self.camera_configs:
                cam_data["pipe"].wait_for_frames(2000)

        print("All cameras locked and ready.")

    def __len__(self): return 9999999 

    def flush_buffers(self):
        """Aggressively drains the default buffer queue for all cameras."""
        for cam_data in self.camera_configs:
            pipe = cam_data["pipe"]
            while True:
                frames = pipe.poll_for_frames()
                if not frames:
                    break # Queue is completely empty

    def close(self):
        """Explicitly stop all RealSense pipelines to free the USB devices."""
        print("Stopping camera pipelines...")
        for cam_data in self.camera_configs:
            try:
                cam_data["pipe"].stop()
            except Exception as e:
                pass
        print("Cameras released.")

    def _get_views(self, idx, num_views, resolution):
        views = []
        self.flush_buffers()
        pose_data = self.base_client.GetMeasuredCartesianPose()
        t_ee2base = np.array([pose_data.x, pose_data.y, pose_data.z])
        r_ee2base = R.from_euler('xyz', [pose_data.theta_x, pose_data.theta_y, pose_data.theta_z], degrees=True).as_matrix()
        
        T_base_to_ee = np.eye(4)
        T_base_to_ee[:3, :3] = r_ee2base
        T_base_to_ee[:3, 3] = t_ee2base
        
        T_base_to_eih = T_base_to_ee @ self.T_ee_to_eih
        T_eih_to_base = np.linalg.inv(T_base_to_eih) # For calculating relative poses
        
        for cam_data in self.camera_configs:
            serial = cam_data["serial"]
            if serial == self.eye_in_hand_serial:
                cam_pose_base = T_base_to_eih
            else:
                cam_pose_base = cam_data["static_pose"]
            cam_pose_relative = T_eih_to_base @ cam_pose_base

            try:
                frames = cam_data["pipe"].wait_for_frames(timeout_ms=1000)
                aligned = cam_data["align"].process(frames)
                
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    print(f"⚠️ Warning: Incomplete frames dropped for camera {serial}")
                    continue

                img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * cam_data["depth_scale"]
                
            except Exception as e:
                print(f"❌ ERROR: Failed to fetch frame from camera {serial}. Error: {e}")
                continue

            img_p, depth_p, intr_p, _ = self._crop_resize_if_necessary(
                image=img, resolution=resolution, depthmap=depth,
                intrinsics=cam_data["K"].astype(np.float32), additional_quantities={}
            )

            views.append({
                "img_raw": img,                   
                "img": img_p,                     
                "depthmap": depth_p,              
                "camera_intrinsics": intr_p,      
                "camera_pose": cam_pose_relative.astype(np.float32), 
                "view0_to_base": T_base_to_eih.astype(np.float32),   
                "dataset": "LiveRealsense",
                "label": "robot_workspace",
                "instance": str(idx) 
            })
            
        return views
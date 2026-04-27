'''
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 
python scripts/run_aston_benchmark_grasp.py --data_root /media/kieran/Extreme_SSD/data/Aston_grasp/scenes/ --camera realsense --model_path /path/to/student_model.pth --num_views 1
'''

import torch
import numpy as np
import argparse
import os
import sys
import threading
import time
import cv2
from scipy.spatial.transform import Rotation as R
import pdb
# Rerun for basic visualization (stripped down)
import rerun as rr

# Custom Kinova & Grasping Utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from mapanything.datasets.custom_live import LiveRealSenseDataset

infer_dir = os.path.abspath("infer") 
if infer_dir not in sys.path:
    sys.path.insert(0, infer_dir)
    
import utilities
from robotic_grasping_utilities import (RobotController, parse_joints_from_json,
                                        RobotModel, Gripper, Pose6D, PathPlanner, 
                                        calculate_pre_grasp_matrix, MODES)

from ultralytics import YOLO
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

# ==============================================================================
# INTEGRATED STUDENT MODEL FUNCTIONS
# ==============================================================================
def load_student_model(model_path, device):
    print(f"Loading student YOLO model from {model_path}...")
    # The student code uses ultralytics YOLO
    # We load it here once, so it doesn't reload every frame
    model = YOLO(model_path) 
    model.to(device) # Optional: uncomment if you want to force YOLO onto a specific device
    return model

def get_3dof_grasp_from_model(model, image, depth, intrinsics, extrinsics):
    """
    Runs the student's YOLO model, logs the 2D bounding boxes to Rerun, 
    and converts the pixel depth into Kinova base coordinates.
    """
    CONFIDENCE_THRESHOLD = 0.5  
    
    # 1. UN-NORMALIZE THE DINOv2 TENSOR FOR YOLO
    # Set up the standard DINOv2/ImageNet math
    IMG_STD = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    
    # Revert back to 0-255 integers and convert to a NumPy array [H, W, 3]
    img_rgb = ((image * IMG_STD + IMG_MEAN).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # YOLO natively expects BGR format (the OpenCV standard)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 2. Run inference on the clean, unnormalized BGR image
    results = model(img_bgr, conf=CONFIDENCE_THRESHOLD)
    
    # === 2D VISUALIZATION ===
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    rr.log("yolo/annotated_feed", rr.Image(annotated_rgb))
    # ========================
    
    detections = results[0].boxes
    
    if len(detections) == 0:
        print("YOLO: No objects detected.")
        return None, None, None, None, None

    # Grab the first detected object
    box = detections[0]
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    
    # 3. Get Center Coordinates
    cx, cy, w, h = box.xywh[0].tolist()
    ix, iy = int(cx), int(cy)
    
    # 4. Get Z Depth
    if len(depth.shape) == 3:
        distance_z = float(depth[iy, ix, 0])
    else:
        distance_z = float(depth[iy, ix])    
    if distance_z <= 0:
        print(f"YOLO: Detected {class_name}, but depth is invalid (0.0) at center.")
        return None, None, None, None, None
    
    # 5. Deproject Pixel to 3D Camera Coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    px, py = intrinsics[0, 2], intrinsics[1, 2]
    
    cam_x = (ix - px) * distance_z / fx
    cam_y = (iy - py) * distance_z / fy
    cam_z = distance_z
    
    print(f"YOLO Found {class_name} | Cam Frame (m): X:{cam_x:.2f}, Y:{cam_y:.2f}, Z:{cam_z:.2f}")

    # 6. Transform Camera Coordinates to Robot Base Coordinates
    cam_point_homog = np.array([cam_x, cam_y, cam_z, 1.0])
    base_point_homog = extrinsics @ cam_point_homog
    
    target_x = base_point_homog[0]
    target_y = base_point_homog[1]
    target_z = base_point_homog[2]

    # 7. Yaw Orientation
    target_yaw_rad = 0.0 
    
    return target_x, target_y, target_z, target_yaw_rad, class_id

# Simplified Constants
MAX_WIDTH, TCP_OFFSET, PRE_GRASP_DISTANCE = 0.14, 0.03, 0.10
joint_limits = [(-154.1, 154.1), (-150.1, 150.1), (-150.1, 150.1), (-148.98, 148.98), (-144.97, 145.0), (-148.98, 148.98)]
latest_views = None

def run_live_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rr.init("Grasp_3DOF", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    print(f"Using device: {device}")

    # 1. Load the new model
    student_model = load_student_model(args.model_path, device)

    # 2. Setup Rerun (Basic feed only)
    rr.init("GRASP3R_3DOF", spawn=True)
    
    print("Connecting to Kinova arm...")
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base_client = BaseClient(router)
        robot = RobotController(router)
        gripper = Gripper(router)
        robot_model = RobotModel(router)

        # Ensure we are only fetching 1 view
        args.num_views = 1 
        dataset = LiveRealSenseDataset(
            base_client=base_client,  
            gripper=gripper,  
            router=router,
            base_dir=args.data_root, 
            split="test",
            resolution=(640, 640),
            # resolution=(518, 518),
            data_norm_type="dinov2",
            transform="imgnorm",
            num_views=args.num_views
        )    

        # 3. Background Camera Thread
        def live_camera_stream():
            global latest_views
            live_frame_idx = 0
            TARGET_HEIGHT = 480 
            
            while True:
                views_raw = dataset[0] 
                latest_views = views_raw 
                
                if len(views_raw) > 0:
                    rr.set_time("live_frame", sequence=live_frame_idx)
                    v_idx = 0
                    if "img_raw" in views_raw[v_idx]:
                        raw_img = np.ascontiguousarray(views_raw[v_idx]["img_raw"])
                        h, w = raw_img.shape[:2]
                        if h > TARGET_HEIGHT:
                            target_width = int((TARGET_HEIGHT / h) * w)
                            display_img = cv2.resize(raw_img, (target_width, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                        else:
                            display_img = raw_img
                        rr.log(f"live_feeds/camera_{v_idx}/image", rr.Image(display_img))
                            
                live_frame_idx += 1
                time.sleep(0.01)

        cam_thread = threading.Thread(target=live_camera_stream, daemon=True)
        cam_thread.start()

        # Load standard poses
        bin_can_pose = parse_joints_from_json('infer/Bin_alcan.json') if os.path.exists('infer/Bin_alcan.json') else None
        bin_plastic_pose = parse_joints_from_json('infer/Bin_plasticb.json') if os.path.exists('infer/Bin_plasticb.json') else None
        before_bin_location_path = 'infer/before_bin.json'
        before_bin_pose = parse_joints_from_json(before_bin_location_path) if os.path.exists(before_bin_location_path) else None

        print("\nStarting simple 3DOF live inference loop.")
        
        with torch.inference_mode():
            while True:
                if latest_views is None or len(latest_views) == 0:
                    time.sleep(0.1) 
                    continue
                
                # Grab the first camera's view
                view_0 = latest_views[0]
                
                # Open gripper and prepare
                gripper.open()
                start_config = robot_model.get_current_joint_configuration()
                
                # Extract image and camera data (Adapt keys to what dataset provides)
                img = view_0.get("img_raw")
                img_view = view_0.get("img")
                
                # Check for depth, with a fallback key just in case
                depth = view_0.get("depth_raw")
                if depth is None:
                    depth = view_0.get("depthmap")
                
                intrinsics = view_0.get("camera_intrinsics")
                extrinsics = view_0.get("view0_to_base") 

                # === FULL 3D POINT CLOUD VISUALIZATION ===
                # Safety check: Only run if the camera successfully captured depth this frame
                if depth is not None and img is not None:
                    h, w, _ = depth.shape
                    
                    # No downsampling! Grabbing every single pixel.
                    v, u = np.mgrid[0:h, 0:w]
                    
                    # Filter out bad depth values
                    d = depth[v, u, 0]
                    valid = (d > 0.05) & (d < 2.0) # Only keep depth between 5cm and 2m
                    
                    if np.any(valid):
                        u_valid, v_valid, z_cam = u[valid], v[valid], d[valid]
                        
                        # Vectorized deprojection to camera coordinates
                        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                        px, py = intrinsics[0, 2], intrinsics[1, 2]
                        x_cam = (u_valid - px) * z_cam / fx
                        y_cam = (v_valid - py) * z_cam / fy
                        
                        # Transform to Robot Base coordinates
                        pts_cam_homog = np.stack((x_cam, y_cam, z_cam, np.ones_like(z_cam)), axis=-1)
                        pts_base = (extrinsics @ pts_cam_homog.T).T[:, :3]
                        
                        # Log full point cloud to Rerun!
                        img_np = ((img_view * IMG_STD + IMG_MEAN).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        colors = img_np[v_valid, u_valid]
                        rr.log("world/scene/point_cloud", rr.Points3D(pts_base, colors=colors, radii=0.002))
                else:
                    # Silently skip visualization if the frame is missing depth data
                    pass
                # ==============================================
                # ==========================================
                # GET GRASP FROM NEW STUDENT MODEL
                # ==========================================
                grasp_x, grasp_y, grasp_z, grasp_yaw, class_id = get_3dof_grasp_from_model(student_model, img_view, depth, intrinsics, extrinsics)
                
                if grasp_x is None:
                    time.sleep(0.5)
                    continue

                # Convert 3DOF (X, Y, Z, Yaw) into a 6DOF top-down grasp matrix
                # Top-down means we point the Z-axis of the gripper down into the table.
                # A standard top-down approach usually involves Roll=180, Pitch=0, Yaw=predicted_yaw
                rot = R.from_euler('xyz', [180, 0, np.rad2deg(grasp_yaw)], degrees=True).as_matrix()
                
                T_final_grasp = np.eye(4)
                T_final_grasp[:3, :3] = rot
                T_final_grasp[:3, 3] = [grasp_x, grasp_y, grasp_z - 0.02]
                
                # Offset by TCP (tool center point) to ensure the fingers reach the object, not the wrist
                approach_vector_z = T_final_grasp[:3, 2]
                T_final_grasp[:3, 3] -= (TCP_OFFSET * approach_vector_z)

                # === MINIMAL GRASP VISUALIZATION ===
                rr.log("world/scene/pre_grasp", rr.Transform3D(
                    translation=T_final_grasp[:3, 3],
                    mat3x3=T_final_grasp[:3, :3],
                ))
                # ===================================
                # Calculate Pre-Grasp and Retreat
                T_pre_grasp = calculate_pre_grasp_matrix(T_final_grasp, distance=PRE_GRASP_DISTANCE)
                
                rr.log("world/scene/target_grasp", rr.Transform3D(
                    translation=T_pre_grasp[:3, 3],
                    mat3x3=T_pre_grasp[:3, :3],
                   
                ))
                # Simple straight-up retreat
                T_retreat = np.copy(T_final_grasp)
                T_retreat[2, 3] += 0.35 # Move 15cm straight up
                
                final_grasp_pose_6d = Pose6D.from_matrix(T_final_grasp)
                pre_grasp_pose_6d = Pose6D.from_matrix(T_pre_grasp)
                retreat_pose_6d = Pose6D.from_matrix(T_retreat)

                # ==========================================
                # PLAN AND EXECUTE
                # ==========================================
                # Empty obstacles list since we removed collision detection
                empty_obstacles = np.array([]) 
                planner = PathPlanner(
                    robot_model, 
                    empty_obstacles, 
                    joint_limits, 
                    max_iter=5000,      
                    step_size=5.0,     
                    use_rrt_star=False,
                    robot_urdf=None
                )

                print("\nPlanning paths...")
                ik_pre = robot_model.compute_best_ik_solution(pre_grasp_pose_6d)
                ik_retreat = robot_model.compute_best_ik_solution(retreat_pose_6d)
                
                if not ik_pre or not ik_retreat:
                    print("  - Warning: Invalid IK for grasp. Skipping frame.")
                    time.sleep(1)
                    continue

                path_to_pre_grasp_raw = planner.plan(start_config, ik_pre)

                # Select correct bin based on detected class (0: Can, 2: Plastic)
                if class_id == 0:
                    target_bin_pose = bin_can_pose
                elif class_id == 2:
                    target_bin_pose = bin_plastic_pose
                else:
                    target_bin_pose = None
                
                if not path_to_pre_grasp_raw:
                    print("  - Warning: RRT failed to find a path to pre-grasp. Skipping frame.")
                    time.sleep(1)
                    continue

                path_to_pre_grasp = planner.smooth_path(path_to_pre_grasp_raw)

                user_choice = input(f"\nTarget acquired at X:{grasp_x:.2f}, Y:{grasp_y:.2f}. Execute? [y: yes / n: skip]: ").strip().lower()
                if user_choice != 'y':
                    print("Skipping...")
                    time.sleep(0.5)
                    continue 
                
                print("\nExecuting simple pick-and-place sequence...")
                try:
                    print("   - Moving to pre-grasp position...")
                    robot.execute_trajectory(path_to_pre_grasp)

                    print("   - Executing final linear approach...")
                    object_grasped = robot.move_and_close(final_grasp_pose_6d, speed_mps=0.1, MAX_OPENING_M=MAX_WIDTH, target_width_m=MAX_WIDTH)
                    gripper.close()
                    print("   - Extracting object safely...")
                    robot.execute_linear_move(retreat_pose_6d, speed_mps=0.2)

                    if object_grasped and before_bin_pose and target_bin_pose:
                        path_to_before_bin_raw = planner.plan(robot_model.get_current_joint_configuration(), before_bin_pose[0])
                        path_to_bin_raw = planner.plan(before_bin_pose[0], target_bin_pose[0]) if path_to_before_bin_raw else None

                        if path_to_before_bin_raw and path_to_bin_raw:
                            print("   - Moving to bin...")
                            full_bin_path = planner.smooth_path(path_to_before_bin_raw + path_to_bin_raw[1:])
                            robot.execute_trajectory(full_bin_path)

                        print("   - Dropping object...")
                        gripper.open()
                        
                        print("   - Returning to start...")
                        path_to_start = planner.smooth_path(planner.plan(robot_model.get_current_joint_configuration(), start_config))
                        robot.execute_trajectory(path_to_start)

                except Exception as e:
                    print(f"An error occurred during execution: {e}")
                    gripper.open()
                    robot.execute_linear_move(pre_grasp_pose_6d, speed_mps=0.1)
                
                finally:
                    print("Execution finished. Waiting for arm to settle...")
                    time.sleep(1.5) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to your calib_data folder")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_views", type=int, default=1)
    args = utilities.parseConnectionArguments(parser)
    
    run_live_inference(args)
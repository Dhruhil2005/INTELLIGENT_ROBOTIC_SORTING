import numpy as np
import torch
import time
import pdb
import os 
from omegaconf import OmegaConf
import rerun as rr
import yourdfpy
import json 
from scipy.spatial.transform import Rotation
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.Exceptions.KServerException import KServerException
import math 
import ikpy.chain
from scipy.spatial import cKDTree 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

KEYS_RGB_ONLY = ["img", "is_metric_scale", "data_norm_type", "dataset", "label", "instance", "idx", "rng", "true_shape"]
KEYS_K = ["ray_directions_cam"]
KEYS_POSE = ["camera_pose_quats", "camera_pose_trans"]
KEYS_DEPTH = ["depth_along_ray"]

MODES = {
    "RGB": {
        "whitelist": KEYS_RGB_ONLY
    },
    "RGB_Intrinsics": {
        "whitelist": KEYS_RGB_ONLY + KEYS_K
    },
    "RGB_Intrinsics_Pose": {
        "whitelist": KEYS_RGB_ONLY + KEYS_K + KEYS_POSE
    },
    "RGB_Intrinsics_Depth": {
        "whitelist": KEYS_RGB_ONLY + KEYS_K + KEYS_DEPTH
    },
    "RGB_Intrinsics_Pose_Depth": {
        "whitelist": KEYS_RGB_ONLY + KEYS_K + KEYS_POSE + KEYS_DEPTH
    }
}

def is_path_clear_upwards(pts_tensor, grasp_matrix, lift_distance=0.1, radius=0.06, noise_threshold=15):
    """
    Checks if a vertical cylinder above the grasp pose is free of obstacles.
    """
    # Extract the X, Y, Z of the grasp in world space
    grasp_pos = torch.tensor(grasp_matrix[:3, 3], device=pts_tensor.device, dtype=torch.float32)

    # 1. Z-Mask: Fast filter to only look at points strictly ABOVE the grasp
    # up to our intended lift distance.
    z_mask = (pts_tensor[:, 2] > grasp_pos[2]) & (pts_tensor[:, 2] < (grasp_pos[2] + lift_distance))

    if not z_mask.any():
        return True # The airspace above is completely empty

    # 2. XY-Mask: Of the points above us, are they inside our vertical cylinder?
    pts_above = pts_tensor[z_mask]
    
    # Calculate squared distance from the vertical center line to avoid heavy torch.sqrt()
    xy_dist_sq = (pts_above[:, 0] - grasp_pos[0])**2 + (pts_above[:, 1] - grasp_pos[1])**2
    
    # Count how many points fall inside the cylinder radius
    points_in_cylinder = (xy_dist_sq < radius**2).sum().item()
    
    # Return True if it's clear (allowing a few points for camera noise)
    return points_in_cylinder <= noise_threshold


def plot_path_to_rerun(path, name, color, robot_urdf):
    ee_path_points = []
    for config_ in path:
        joint_dict = {f"J{j}": np.deg2rad(angle) for j, angle in enumerate(config_.angles)}
        joint_dict.update({
            "right_finger_bottom_joint": 0.0, "right_finger_tip_joint": 0.0,
            "left_finger_bottom_joint": 0.0, "left_finger_tip_joint": 0.0
        })
        robot_urdf.update_cfg(joint_dict)
        T_EE = robot_urdf.get_transform('END_EFFECTOR', 'BASE')
        ee_path_points.append(T_EE[:3, 3])
    rr.log(name, rr.LineStrips3D([ee_path_points], colors=[color], radii=[0.005]))


def voxel_down_sample_torch_with_colors(points, colors, voxel_size):
    """
    Voxel Subsampling (Point & Color Preservation).
    """
    valid_mask = ~torch.isnan(points).any(dim=1)
    points = points[valid_mask]
    colors = colors[valid_mask] # Keep colors synced
    quantized = torch.floor(points.float() / voxel_size).long()
    x, y, z = quantized.unbind(1)
    hashed = x * 73856093 + y * 19349663 + z * 83492791
    unique_hashes, inverse_indices = torch.unique(hashed, sorted=True, return_inverse=True)
    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
    num_voxels = unique_hashes.size(0)

    selector = torch.full((num_voxels,), fill_value=len(points), device=points.device, dtype=torch.long)
    selector.scatter_reduce_(0, inverse_indices, perm, reduce="amin", include_self=False)
    downsampled_points = points[selector]
    downsampled_colors = colors[selector]
    
    # return downsampled_points, downsampled_colors
    return downsampled_points.half(), downsampled_colors

def log_ik_attempt(name, ik_solution, target_matrix, start_angles):
    if ik_solution is None:
        print(f"[IK FAIL] {name}")
        return None
    
    angles = np.array(ik_solution.angles)
    dist = np.linalg.norm(angles - start_angles)

    print(f"[IK SUCCESS] {name}")
    print(f"  joint angles: {np.round(angles, 3)}")
    print(f"  joint distance from start: {dist:.3f}")

    pos = target_matrix[:3,3]
    print(f"  target position: {np.round(pos,4)}")

    return dist

def log_grasp_group_to_rerun(grasp_group, base_path, start_idx=0):
    translations = grasp_group.translations.detach().cpu().numpy() if hasattr(grasp_group.translations, "cpu") else grasp_group.translations
    rotations = grasp_group.rotation_matrices.detach().cpu().numpy() if hasattr(grasp_group.rotation_matrices, "cpu") else grasp_group.rotation_matrices
    widths = grasp_group.widths.detach().cpu().numpy() if hasattr(grasp_group.widths, "cpu") else grasp_group.widths
    depths = grasp_group.depths.detach().cpu().numpy() if hasattr(grasp_group.depths, "cpu") else grasp_group.depths
    scores = grasp_group.scores.detach().cpu().numpy() if hasattr(grasp_group.scores, "cpu") else grasp_group.scores

    # 2. Now iterate through the fast NumPy arrays
    for i in range(len(grasp_group)):
        T = translations[i]
        R_mat = rotations[i]
        W = float(widths[i])
        D = float(depths[i])
        score = float(scores[i])

        color = [int((1 - score) * 255), int(score * 255), 0]
        entity_path = f"{base_path}/{start_idx + i}"
        
        rr.log(entity_path, rr.Transform3D(translation=T, mat3x3=R_mat))

        hw = W / 2.0
        finger_thk = 0.005
        palm_depth = 0.015

        c_left  = [D/2, -(hw + finger_thk/2), 0]
        c_right = [D/2,  (hw + finger_thk/2), 0]
        
        c_palm  = [-palm_depth/2, 0, 0]

        s_left  = [D, finger_thk, finger_thk]
        s_right = [D, finger_thk, finger_thk]
        s_palm  = [palm_depth, W + finger_thk*4, finger_thk*3]

        rr.log(
            f"{entity_path}/geometry",
            rr.Boxes3D(
                centers=[c_left, c_right, c_palm],
                sizes=[s_left, s_right, s_palm],
                colors=color,
                fill_mode="solid"
            )
        )
    return start_idx + len(grasp_group)

def special_float_resolver(x): return float(x)
if not OmegaConf.has_resolver("special_float"):
    OmegaConf.register_new_resolver("special_float", special_float_resolver)

def fit_plane_ransac_torch(pts_tensor, distance_threshold=0.01, num_iterations=1000):
    """Vectorized RANSAC plane fitting purely in PyTorch."""
    N = pts_tensor.shape[0]
    if N < 3: return None, None
    idx = torch.randint(0, N, (num_iterations, 3), device=pts_tensor.device)
    samples = pts_tensor[idx] 
    p0, p1, p2 = samples[:, 0, :], samples[:, 1, :], samples[:, 2, :]
    normals = torch.nn.functional.normalize(torch.cross(p1 - p0, p2 - p0, dim=1), p=2, dim=1)
    d = -torch.sum(normals * p0, dim=1, keepdim=True) 
    distances = torch.abs(torch.matmul(pts_tensor, normals.T) + d.T)
    inliers_mask = distances < distance_threshold 
    best_iter = torch.argmax(inliers_mask.sum(dim=0))
    
    return normals[best_iter], inliers_mask[:, best_iter]

# def get_obb_mask(pts, roi_center_t, inv_roi_rot_matrix, half_extents, max_radius=0.7):
#     local_pts = (pts - roi_center_t) @ inv_roi_rot_matrix.T
#     obb_mask = (torch.abs(local_pts) <= half_extents).all(dim=-1)
#     cyl_mask = (pts[..., 0]**2 + pts[..., 1]**2) <= (max_radius**2)
#     return obb_mask & cyl_mask

def get_obb_mask(pts, roi_center_t, inv_roi_rot_matrix, half_extents, max_radius=0.7):
    local_pts = (pts - roi_center_t) @ inv_roi_rot_matrix.T
    obb_mask = (torch.abs(local_pts) <= half_extents).all(dim=-1)
    
    # Create an empty mask for the final result
    final_mask = torch.zeros_like(obb_mask)
    
    # Grab only the points that passed the OBB check
    valid_pts = pts[obb_mask]
    
    # Only do the expensive square math if points actually survived
    if len(valid_pts) > 0:
        cyl_mask = (valid_pts[..., 0]**2 + valid_pts[..., 1]**2) <= (max_radius**2)
        final_mask[obb_mask] = cyl_mask
        
    return final_mask

def setup_urdf_meshes_in_rerun(urdf_path, base_entity_path):
    """Parses a URDF and logs meshes and primitive geometries statically."""
    print(f"Loading URDF geometries into Rerun from {urdf_path}...")
    try:
        urdf = yourdfpy.URDF.load(urdf_path)
        
        for link in urdf.robot.links:
            if link.visuals:
                geom = link.visuals[0].geometry
                entity_path = f"{base_entity_path}/{link.name}"
                
                # 1. Handle External Meshes (.STL, .OBJ)
                if geom.mesh:
                    mesh_path = geom.mesh.filename
                    if os.path.exists(mesh_path):
                        rr.log(entity_path, rr.Asset3D(path=mesh_path), static=True)
                    else:
                        print(f"Warning: Mesh not found at {mesh_path}")
                        
                # 2. Handle URDF Box Primitives
                elif geom.box:
                    # Rerun takes half-sizes for boxes
                    hx, hy, hz = [s / 2.0 for s in geom.box.size]
                    rr.log(entity_path, rr.Boxes3D(half_sizes=[[hx, hy, hz]], colors=[[150, 150, 150]]), static=True)
                    
                # 3. Handle URDF Cylinder Primitives
                elif geom.cylinder:
                    rr.log(entity_path, rr.Cylinders3D(radii=[geom.cylinder.radius], lengths=[geom.cylinder.length], colors=[[50, 50, 50]]), static=True)

        return urdf
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        return None

class PathPlanner:
    """
    An optimized RRT/RRT* path planner for collision-free motion planning.
    """
    class Node:
        def __init__(self, config):
            self.config = config
            self.parent = None
            self.cost = 0.0

    def __init__(self, robot_model, obstacles, joint_limits, max_iter=5000, step_size=5.0, goal_bias=0.1, rewire_radius=15.0, use_rrt_star=False, robot_urdf=False):
        self.robot_model = robot_model
        self.obstacles = np.array(obstacles)
        self.joint_limits = joint_limits
        
        if not isinstance(joint_limits, list) or not all(isinstance(i, tuple) and len(i) == 2 for i in joint_limits):
            raise ValueError("`joint_limits` must be a list of tuples, e.g., [(-180, 180), ...]")
        
        self.dof = len(self.joint_limits)
        print(f"PathPlanner initialized for a robot with {self.dof} degrees of freedom.")

        self.max_iter = max_iter
        self.robot_urdf = robot_urdf
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.use_rrt_star = use_rrt_star # Toggle for faster standard RRT
        self.tree = []
        
        # Pre-allocate numpy array for lightning-fast nearest neighbor searches
        self.tree_configs = np.zeros((self.max_iter + 2, self.dof))
        
        if self.obstacles.any():
            self.obstacle_kdtree = cKDTree(self.obstacles) # Use C-optimized KDTree
        else:
            self.obstacle_kdtree = None
            
        # self.collision_spheres = [('end_effector', 0.15)]
        self.collision_spheres = [
            ('SHOULDER', 0.09),
            ('ARM', 0.08),
            ('FOREARM', 0.07),
            ('LOWER_WRIST', 0.06),
            ('UPPER_WRIST', 0.06),
            ('END_EFFECTOR', 0.05),
            ('gripper_base_link', 0.05),
            ('camera_mount_link', 0.07),
            ('d455_link', 0.07)
        ]
        self.joint_names = [f"J{i}" for i in range(self.dof)]
        self.gripper_defaults = {
            "right_finger_bottom_joint": 0.0, "right_finger_tip_joint": 0.0,
            "left_finger_bottom_joint": 0.0, "left_finger_tip_joint": 0.0
        }

    def is_colliding(self, config):
        if self.obstacle_kdtree is None:
            return False

        joint_dict = dict(zip(self.joint_names, np.deg2rad(config.angles)))
        joint_dict.update(self.gripper_defaults)
        
        self.robot_urdf.update_cfg(joint_dict)
        for link_name, radius in self.collision_spheres:

            T_link = self.robot_urdf.get_transform(link_name, 'BASE')
            link_pos = T_link[:3, 3]

            # indices = self.obstacle_kdtree.query_ball_point(link_pos, r=radius + 0.02)
            # if len(indices) > 0:
            #     return True     
            dist, _ = self.obstacle_kdtree.query(link_pos, k=1)

            if dist <= (radius + 0.02):
                return True           

        return False

    def plan(self, start_config, goal_config):
        """Plans a path from a start to a goal joint configuration."""
        self.tree = [self.Node(start_config)]
        self.tree_configs[0] = start_config.angles
        
        for i in range(self.max_iter):
            rnd_node = self.get_random_node(goal_config)
            nearest_node = self.find_nearest(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if new_node and not self.is_collision_segment(nearest_node.config, new_node.config):
                
                if self.use_rrt_star:
                    near_indices = self.find_near_nodes(new_node)
                    self.choose_parent(new_node, near_indices)
                    self.add_to_tree(new_node)
                    self.rewire(new_node, near_indices)
                else:
                    # Standard RRT is much faster. Just attach to the nearest node.
                    self.add_to_tree(new_node)
            
            # Check if we are close enough to the goal
            if self.tree and self.distance(self.tree[-1].config, goal_config) < self.step_size:
                final_node = self.Node(goal_config)
                if not self.is_collision_segment(self.tree[-1].config, final_node.config):
                    final_node.parent = self.tree[-1]
                    print(f"Goal reached by planner after {i} iterations.")
                    return self.generate_path(final_node)
        
        print(f"Planner failed to find a path after {self.max_iter} iterations.")
        return None

    def add_to_tree(self, node):
        """Helper to keep the numpy tracking array in sync with the tree list."""
        idx = len(self.tree)
        self.tree.append(node)
        self.tree_configs[idx] = node.config.angles

    def get_random_node(self, goal_config):
        if np.random.rand() > self.goal_bias:
            rnd_angles = [np.random.uniform(low, high) for low, high in self.joint_limits]
            return self.Node(type(goal_config)(rnd_angles)) 
        else:
            return self.Node(goal_config)

    def get_angular_diff(self, a, b):
        # Removed modulo wrap. Bounded joints use standard linear difference.
        return a - b

    def find_nearest(self, target_node):
        """Vectorized nearest neighbor search."""
        target_angles = np.array(target_node.config.angles)
        active_nodes = len(self.tree)
        distances = np.linalg.norm(self.tree_configs[:active_nodes] - target_angles, axis=1)
        min_index = np.argmin(distances)
        return self.tree[min_index]

    def steer(self, from_node, to_node):
        from_angles = np.array(from_node.config.angles)
        to_angles = np.array(to_node.config.angles)
        
        diffs = self.get_angular_diff(to_angles, from_angles)
        dist = np.linalg.norm(diffs)
        
        if dist < 1e-6:
            return None

        if dist < self.step_size:
            new_angles = from_angles + diffs
            cost_addition = dist
        else:
            step_vector = self.step_size * (diffs / dist)
            new_angles = from_angles + step_vector
            cost_addition = self.step_size

        clamped_angles = []
        for i, angle in enumerate(new_angles):
            low, high = self.joint_limits[i]
            clamped_angles.append(float(np.clip(angle, low, high)))

        new_node = self.Node(type(to_node.config)(clamped_angles))
        new_node.cost = from_node.cost + cost_addition
        new_node.parent = from_node
        return new_node

    def is_collision_segment(self, from_config, to_config, num_checks=4):
        from_angles = np.array(from_config.angles)
        to_angles = np.array(to_config.angles)
        
        for i in range(num_checks, 0, -1):
            t = i / num_checks
            interp_angles = from_angles + t * (to_angles - from_angles)
            if self.is_colliding(type(from_config)(interp_angles.tolist())):
                return True
        return False

    def distance(self, config1, config2):
        arr1 = np.array(config1.angles)
        arr2 = np.array(config2.angles)
        return np.linalg.norm(arr1 - arr2)

    def find_near_nodes(self, new_node):
        """Vectorized search for near nodes."""
        target_angles = np.array(new_node.config.angles)
        active_nodes = len(self.tree)
        distances = np.linalg.norm(self.tree_configs[:active_nodes] - target_angles, axis=1)
        return np.where(distances <= self.rewire_radius)[0]

    def choose_parent(self, new_node, near_indices):
        min_cost = new_node.cost
        best_parent = new_node.parent

        for i in near_indices:
            near_node = self.tree[i]
            dist_to_near = self.distance(near_node.config, new_node.config)
            cost = near_node.cost + dist_to_near
            if cost < min_cost and not self.is_collision_segment(near_node.config, new_node.config):
                min_cost = cost
                best_parent = near_node
        
        new_node.parent = best_parent
        new_node.cost = min_cost

    def rewire(self, new_node, near_indices):
        for i in near_indices:
            near_node = self.tree[i]
            cost = new_node.cost + self.distance(new_node.config, near_node.config)
            if cost < near_node.cost and not self.is_collision_segment(new_node.config, near_node.config):
                near_node.parent = new_node
                near_node.cost = cost
    
    def generate_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]
    
    def smooth_path(self, path, iterations=100):
        if not path or len(path) < 3:
            return path
        smoothed_path = list(path)
        for _ in range(iterations):
            num_waypoints = len(smoothed_path)
            if num_waypoints < 3: break
            idx1, idx2 = sorted(np.random.choice(num_waypoints, 2, replace=False))
            if idx2 == idx1 + 1: continue
            config1, config2 = smoothed_path[idx1], smoothed_path[idx2]
            if not self.is_collision_segment(config1, config2):
                del smoothed_path[idx1 + 1:idx2]
        return smoothed_path


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

class Pose6D:
    # A simple container for a 6D pose.
    def __init__(self, x, y, z, theta_x, theta_y, theta_z):
        self.x = x
        self.y = y
        self.z = z
        self.theta_x = theta_x  # Degrees
        self.theta_y = theta_y  # Degrees
        self.theta_z = theta_z  # Degrees

    def __repr__(self):
        # A nice string representation for printing
        return (f"Pose6D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, "
                f"tx={self.theta_x:.2f}, ty={self.theta_y:.2f}, tz={self.theta_z:.2f})")

    @staticmethod
    def from_kortex_pose(kortex_pose):
        return Pose6D(kortex_pose.x, kortex_pose.y, kortex_pose.z, 
                      kortex_pose.theta_x, kortex_pose.theta_y, kortex_pose.theta_z)

    @classmethod
    def from_matrix(cls, T):
        """Creates a Pose6D object from a 4x4 transformation matrix."""
        x = T[0, 3]
        y = T[1, 3]
        z = T[2, 3]

        # Extract the rotation matrix
        R = T[:3, :3]

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            theta_x_rad = np.arctan2(R[2, 1], R[2, 2])
            theta_y_rad = np.arctan2(-R[2, 0], sy)
            theta_z_rad = np.arctan2(R[1, 0], R[0, 0])
        else:
            theta_x_rad = np.arctan2(-R[1, 2], R[1, 1])
            theta_y_rad = np.arctan2(-R[2, 0], sy)
            theta_z_rad = 0
            
        theta_x = np.rad2deg(theta_x_rad)
        theta_y = np.rad2deg(theta_y_rad)
        theta_z = np.rad2deg(theta_z_rad)
        
        return cls(x, y, z, theta_x, theta_y, theta_z)

    def to_matrix(self):
        """Converts the Pose6D object back to a 4x4 transformation matrix."""
        T = np.eye(4)
        # Set the translation component
        T[0:3, 3] = [self.x, self.y, self.z]
        r = Rotation.from_euler('xyz', [self.theta_x, self.theta_y, self.theta_z], degrees=True)
        T[0:3, 0:3] = r.as_matrix()
        return T

def parse_joints_from_json(file_path):
    """
    Loads a JSON file, extracts the actuator positions, normalizes them,
    and returns a JointConfiguration object.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    actuator_data = next((item for item in data if item["title"] == "actuators"), None)
    if not actuator_data:
        raise ValueError(f"Could not find 'actuators' data in {file_path}")

    pos_header_index = -1
    for i, item in enumerate(actuator_data['data']):
        if item.get("isHeaderRow") and item.get("title") == "position":
            pos_header_index = i
            break
    
    if pos_header_index == -1:
        raise ValueError("Could not find position data in actuators section")

    raw_angles = [parse_value(actuator_data['data'][i]['value']) for i in range(pos_header_index + 1, pos_header_index + 7)]
    normalized_angles = [normalize_angle(angle) for angle in raw_angles]
    
    return JointConfiguration(normalized_angles), normalized_angles

class JointConfiguration:
    """Represents the angles of all robot joints."""
    def __init__(self, angles):
        self.angles = angles

def normalize_angle(angle_deg):
    """Normalize an angle from a 0-360 range to the [-180, 180] range."""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg

def unnormalize_angle(angle_deg):
    """Convert a normalized angle from [-180, 180] back to the robot's 0-360 range."""
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def parse_value(value_str: str) -> float:
    """Extracts the floating-point number from a string like '333.977 °'."""
    return float(value_str.split()[0])

class RobotController:
    """
    Handles sending commands to the robot, including trajectory execution.
    """
    def __init__(self, router, timeout_duration=200):
        self.base = BaseClient(router)
        self.timeout = timeout_duration

    def get_current_pose_matrix(self):
        """
        Gets the current end-effector pose from the robot and returns it as a 4x4 numpy matrix.
        """
        feedback = self.base.GetMeasuredCartesianPose()
        pose_6d = Pose6D.from_kortex_pose(feedback)
        return pose_6d.to_matrix()
    
    def _check_for_end_or_abort(self, e):
        """Returns a closure checking for END or ABORT notifications."""
        def check(notification, e=e):
            if notification.action_event == Base_pb2.ACTION_END or \
               notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def execute_trajectory(self, trajectory: list):
        # Ensure the arm is in the correct servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Grab the target configuration
        target_config = trajectory[-1]
        raw_unnormalized_angles = [unnormalize_angle(angle) for angle in target_config.angles]

        # Define Kinova Gen3 lite hard limits (Min, Max)
        JOINT_LIMITS = [
            (-154.1, 154.1),   # Joint 1
            (-150.1, 150.1),   # Joint 2
            (-150.1, 150.1),   # Joint 3
            (-148.98, 148.98), # Joint 4
            (-144.97, 145.0),  # Joint 5
            (-148.98, 148.98)  # Joint 6
        ]

        target_angles = []
        for i, angle in enumerate(raw_unnormalized_angles):
            # 1. Force the angle into the -180 to 180 domain
            angle_180_domain = (angle + 180) % 360 - 180
            
            # 2. Check against the limits
            min_limit, max_limit = JOINT_LIMITS[i]
            safe_min = min_limit + 0.05
            safe_max = max_limit - 0.05
            
            if angle_180_domain > safe_max:
                print(f"  > Clamping Joint {i+1} from {angle_180_domain:.2f}° down to {safe_max:.2f}°")
                target_angles.append(safe_max)
            elif angle_180_domain < safe_min:
                print(f"  > Clamping Joint {i+1} from {angle_180_domain:.2f}° up to {safe_min:.2f}°")
                target_angles.append(safe_min)
            else:
                target_angles.append(angle_180_domain)

        # 1. Create the Action
        action = Base_pb2.Action()
        action.name = "reach_pose"
        action.application_data = ""

        # 2. Add the clamped target angles to the action
        for i, angle in enumerate(target_angles):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = angle

        print("Executing reach pose action at maximum safe speed...")
        try:
            self.base.ExecuteAction(action)
            time.sleep(0.5) 
            
            # 4. Robust polling loop with a timeout
            start_time = time.time()
            timeout_seconds = 15.0  
            
            while True:
                if (time.time() - start_time) > timeout_seconds:
                    print(f"\nTimeout! The arm failed to reach the target within {timeout_seconds}s.")
                    return False

                current_angles = self.base.GetMeasuredJointAngles().joint_angles
                
                max_error = 0
                for t, c in zip(target_angles, current_angles):
                    diff = abs(t - c.value) % 360
                    circular_error = min(diff, 360 - diff)
                    max_error = max(max_error, circular_error)
                
                if max_error < 1.0: 
                    print("Pose reached.")
                    return True
                    
                time.sleep(0.1)
            
        except Exception as ex:
            print(f"Failed to execute action: {ex}")
            return False

    def execute_linear_move(self, target_pose: Pose6D, speed_mps: float = 0.1):
        """
        Executes a Cartesian movement with a dynamically calculated wait time.

        Args:
            target_pose (Pose6D): The destination pose.
            speed_mps (float): The desired linear speed in meters per second.
        
        Returns:
            bool: True if the action was sent successfully, False otherwise.
        """
        try:
            current_pose = self.base.GetMeasuredCartesianPose()
        except KServerException as ex:
            print(f"ERROR: Failed to get current robot pose: {ex}")
            return False
        target_pose.theta_x = current_pose.theta_x
        target_pose.theta_y = current_pose.theta_y
        target_pose.theta_z = current_pose.theta_z

        dx = target_pose.x - current_pose.x
        dy = target_pose.y - current_pose.y
        dz = target_pose.z - current_pose.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        if distance < 0.001: # 1mm threshold
            print("Target is at the current location. No move needed.")
            return True

        estimated_duration = distance / speed_mps
        wait_duration = estimated_duration + 1.0
        
        print(f"Distance to target: {distance:.3f} m")
        print(f"Estimated move time: {estimated_duration:.2f}s (Waiting for {wait_duration:.2f}s)")

        action = Base_pb2.Action()
        action.name = "Linear Move (Calculated Wait)"
        action.application_data = ""

        cartesian_speed = action.reach_pose.constraint.speed
        cartesian_speed.translation = speed_mps

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = target_pose.x
        cartesian_pose.y = target_pose.y
        cartesian_pose.z = target_pose.z
        cartesian_pose.theta_x = target_pose.theta_x
        cartesian_pose.theta_y = target_pose.theta_y
        cartesian_pose.theta_z = target_pose.theta_z

        print(f"Executing linear move...")
        try:
            self.base.ExecuteAction(action)
            time.sleep(wait_duration)
            print("Linear move completed (based on calculated time).")
            return True
        except KServerException as ex:
            print(f"FAILED to execute action: {ex}")
            return False

    def move_and_close(self, target_pose, target_width_m: float, MAX_OPENING_M: float, speed_mps: float = 0.01) -> bool:
        """
        Executes a Cartesian linear move while proportionally closing the gripper 
        from its current position to the physical `target_width_m`. 
        Fully closes upon arrival.
        """
        print(f"Starting synchronized approach... Target pre-width: {target_width_m:.3f}m")

        clamped_width = max(0.0, min(target_width_m, MAX_OPENING_M))
        target_normalized_pos = 1.0 - (clamped_width / MAX_OPENING_M)

        try:
            current_pose = self.base.GetMeasuredCartesianPose()
        except KServerException as ex:
            print(f"ERROR: Failed to get current robot pose: {ex}")
            return False

        target_pose.theta_x = current_pose.theta_x
        target_pose.theta_y = current_pose.theta_y
        target_pose.theta_z = current_pose.theta_z

        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        try:
            current_gripper_pos = self.base.GetMeasuredGripperMovement(gripper_request).finger[0].value
        except KServerException as ex:
            print(f"ERROR: Failed to get gripper position: {ex}")
            return False

        dx = target_pose.x - current_pose.x
        dy = target_pose.y - current_pose.y
        dz = target_pose.z - current_pose.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        estimated_duration = 0.0
        if distance >= 0.001: 
            estimated_duration = distance / speed_mps
        
        move_wait = estimated_duration + 0.5 

        action = Base_pb2.Action()
        action.name = "Synchronized Linear Move"
        action.reach_pose.constraint.speed.translation = speed_mps
        
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = target_pose.x
        cartesian_pose.y = target_pose.y
        cartesian_pose.z = target_pose.z
        cartesian_pose.theta_x = target_pose.theta_x
        cartesian_pose.theta_y = target_pose.theta_y
        cartesian_pose.theta_z = target_pose.theta_z

        try:
            if distance >= 0.001:
                self.base.ExecuteAction(action)
        except KServerException as ex:
            print(f"FAILED to start arm move: {ex}")
            return False

        sleep_interval = 0.1 
        steps = int(move_wait / sleep_interval)
        
        if steps > 0:
            pos_increment = (target_normalized_pos - current_gripper_pos) / steps
            
            gripper_command = Base_pb2.GripperCommand()
            finger = gripper_command.gripper.finger.add()
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            finger.finger_identifier = 1
            
            for i in range(steps):
                new_pos = current_gripper_pos + (pos_increment * (i + 1))
                finger.value = new_pos
                
                try:
                    self.base.SendGripperCommand(gripper_command)
                except KServerException:
                    pass 
                
                time.sleep(sleep_interval)
        else:
            time.sleep(move_wait)

        print(f"Arrived at target. Executing final full closure from {target_normalized_pos:.3f} to 1.0...")
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = 1.0  # 1.0 is fully closed
        
        try:
            self.base.SendGripperCommand(gripper_command)
        except KServerException as ex:
            print(f"FAILED to execute final grasp: {ex}")
            return False
            
        time.sleep(1.0) # Time to physically clamp

        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        final_position = gripper_measure.finger[0].value
        print(f"Grasp complete. Gripper stopped at position: {final_position:.3f}")
    
        return True

class Gripper:
    """A class to control the gripper."""

    def __init__(self, router):
        self.router = router
        self.base = BaseClient(self.router)

    def open(self, threshold=0.02):
        """Opens the gripper only if it is not already open."""
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        
        try:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            
            if len(gripper_measure.finger) > 0:
                current_position = gripper_measure.finger[0].value
                
                if current_position <= threshold:
                    print(f"Gripper is already open (position: {current_position:.3f}). Skipping.")
                    return  # Exit early! No command sent, no time.sleep()
                    
        except Exception as e:
            print(f"Warning: Could not read gripper state ({e}). Forcing open...")

        print("Opening gripper...")
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = 0.0  # 0.0 is fully open
        self.base.SendGripperCommand(gripper_command)
        
        time.sleep(1)

    def close(self) -> bool:
        """
        Closes the gripper and determines if an object was grasped.
        
        Returns:
            bool: True if an object was grasped, False otherwise.
        """
        print("Closing gripper and checking for object...")
        
        # 1. Command the gripper to close completely
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = 1.0  # 1.0 is fully closed
        self.base.SendGripperCommand(gripper_command)
        time.sleep(1.5) # Allow time for the gripper to close

        # 2. Read the gripper's actual final position
        gripper_request = Base_pb2.GripperRequest()
        
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        final_position = gripper_measure.finger[0].value

        print(f"Gripper stopped at position: {final_position:.3f}")
        return True

    def close_to_width(self, target_width_m: float, MAX_OPENING_M: float) -> bool:
        """
        Moves the gripper to a specific approach width.
        """
        print(f"Moving gripper to target width: {target_width_m:.3f}m...")
        
        clamped_width = max(0.0, min(target_width_m, MAX_OPENING_M))
        
        normalized_position = 1.0 - (clamped_width / MAX_OPENING_M)
        
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = normalized_position 
        
        try:
            self.base.SendGripperCommand(gripper_command)
            
            timeout = 3.0  # Max seconds to wait
            start_time = time.time()
            tolerance = 0.02 # How close is "close enough"
            actual_position = 0.0
            
            while (time.time() - start_time) < timeout:
                gripper_request = Base_pb2.GripperRequest()
                gripper_request.mode = Base_pb2.GRIPPER_POSITION
                gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
                
                if len(gripper_measure.finger) > 0:
                    actual_position = gripper_measure.finger[0].value
                    
                    # If we reached our target, break out of the loop early!
                    if abs(actual_position - normalized_position) <= tolerance:
                        break
                        
                time.sleep(0.1) # Brief pause before checking again
            
            print(f"Gripper reached normalized position: {actual_position:.3f}")
            return True
            
        except Exception as e:
            print(f"Failed to move gripper to width: {e}")
            return False

class RobotModel:
    """
    Handles robot-specific details like kinematics and angle normalization.
    Now supercharged with local IKPy URDF solving.
    """
    def __init__(self, router, urdf_path="infer/gen3_lite_complete.urdf"):
        self.base = BaseClient(router)

        self.ik_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["BASE"],
            active_links_mask=[False, True, True, True, True, True, True, False]
        )
        
        self.active_indices = [i for i, is_active in enumerate(self.ik_chain.active_links_mask) if is_active]
        self.static_fallback_seeds = [
            JointConfiguration([0.0, 15.0, 130.0, 0.0, 50.0, 0.0]),   
            JointConfiguration([0.0, -20.0, 100.0, 0.0, 80.0, 0.0]),  
            JointConfiguration([0.0, 45.0, 90.0, 0.0, 45.0, 0.0]),    
            JointConfiguration([90.0, 15.0, 130.0, 0.0, 50.0, 0.0]),  
            JointConfiguration([-90.0, 15.0, 130.0, 0.0, 50.0, 0.0])  
        ]
        self.Z_ROT_90_MAT = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        self.Z_ROT_180_OBJ = Rotation.from_euler('z', 180, degrees=True)


    def get_current_joint_configuration(self):
        """
        Gets current joint angles directly from the physical robot 
        and NORMALIZES them to the [-180, 180] range.
        """
        feedback = self.base.GetMeasuredJointAngles()
        normalized_angles = [normalize_angle(angle.value) for angle in feedback.joint_angles]
        return JointConfiguration(normalized_angles)

    def get_current_pose_direct(self) -> Pose6D:
        """
        Gets the current end-effector pose directly from the physical robot.
        """
        kortex_pose = self.base.GetMeasuredCartesianPose()
        return Pose6D.from_kortex_pose(kortex_pose)

    def compute_inverse_kinematics(self, target_pose: Pose6D, start_guess_config: JointConfiguration):
        target_matrix = np.eye(4)
        r = Rotation.from_euler('xyz', [target_pose.theta_x, target_pose.theta_y, target_pose.theta_z], degrees=True)
        target_matrix[:3, :3] = r.as_matrix()
        target_matrix[:3, 3] = [target_pose.x, target_pose.y, target_pose.z]

        # urdf_target_matrix = np.copy(target_matrix)
        # urdf_target_matrix[:3, :3] = target_matrix[:3, :3] @ Rotation.from_euler('z', 90, degrees=True).as_matrix()
        urdf_target_matrix = np.eye(4)
        urdf_target_matrix[:3, :3] = target_matrix[:3, :3] @ self.Z_ROT_90_MAT
        urdf_target_matrix[:3, 3] = target_matrix[:3, 3]


        # Build padded array
        initial_position = [0.0] * len(self.ik_chain.links)
        for i, joint_idx in enumerate(self.active_indices):
            if i < len(start_guess_config.angles):
                initial_position[joint_idx] = np.deg2rad(start_guess_config.angles[i])

        ik_solution_rad = self.ik_chain.inverse_kinematics(
            target_position=urdf_target_matrix[:3, 3],
            target_orientation=urdf_target_matrix[:3, :3],
            orientation_mode="all",
            initial_position=initial_position
        )

        # Verify against the URDF target!
        fk_matrix = self.ik_chain.forward_kinematics(ik_solution_rad)
        xyz_error = np.linalg.norm(fk_matrix[:3, 3] - urdf_target_matrix[:3, 3])
        
        if xyz_error > 0.005:
            return None

        ik_solution_deg = [np.rad2deg(ik_solution_rad[idx]) for idx in self.active_indices]
        return JointConfiguration(ik_solution_deg)

    def _create_flipped_pose(self, original_pose: Pose6D) -> Pose6D:
        original_rotation = Rotation.from_euler('xyz', [original_pose.theta_x, original_pose.theta_y, original_pose.theta_z], degrees=True)
        flipped_rotation = original_rotation * self.Z_ROT_180_OBJ        
        # flipped_rotation = original_rotation * flip_rotation
        flipped_euler_angles = flipped_rotation.as_euler('xyz', degrees=True)
        
        return Pose6D(
            x=original_pose.x,
            y=original_pose.y,
            z=original_pose.z,
            theta_x=flipped_euler_angles[0],
            theta_y=flipped_euler_angles[1],
            theta_z=flipped_euler_angles[2]
        )

    @staticmethod
    def _calculate_joint_distance(config1: JointConfiguration, config2: JointConfiguration) -> float:
        """
        Calculates the Euclidean distance between two joint configurations.
        """
        angles1 = np.array(config1.angles)
        angles2 = np.array(config2.angles)
        return np.linalg.norm(angles1 - angles2)

    def compute_best_ik_solution(self, target_pose: Pose6D) -> JointConfiguration:
        """
        Computes IK by testing a target pose (and its Z-flipped version) against 
        a robust list of fallback seed configurations. 
        (Now runs exponentially faster since the inner loop is local).
        """
        current_config = self.get_current_joint_configuration()
        flipped_target_pose = self._create_flipped_pose(target_pose)
        
        active_seeds = [current_config] + self.static_fallback_seeds

        valid_solutions = []

        def solve_with_fallbacks(pose):
            for seed in active_seeds:
                sol = self.compute_inverse_kinematics(pose, seed)
                if sol is not None:
                    valid_solutions.append(sol)
                    break 

        solve_with_fallbacks(target_pose)
        solve_with_fallbacks(flipped_target_pose)

        if not valid_solutions:
            return None

        best_solution = None
        min_dist = float('inf')

        for sol in valid_solutions:
            dist = self._calculate_joint_distance(current_config, sol)
            if dist < min_dist:
                min_dist = dist
                best_solution = sol

        return best_solution

    def compute_forward_kinematics_all_joints(self, config: JointConfiguration) -> Pose6D:
        angles_rad = [0.0] * len(self.ik_chain.links)
        for i, joint_idx in enumerate(self.active_indices):
            if i < len(config.angles):
                angles_rad[joint_idx] = np.deg2rad(config.angles[i])
        
        fk_matrix = self.ik_chain.forward_kinematics(angles_rad)
        
        kortex_matrix = np.copy(fk_matrix)
        kortex_matrix[:3, :3] = fk_matrix[:3, :3] @ Rotation.from_euler('z', -90, degrees=True).as_matrix()
        
        x, y, z = kortex_matrix[:3, 3]
        r = Rotation.from_matrix(kortex_matrix[:3, :3])
        theta_x, theta_y, theta_z = r.as_euler('xyz', degrees=True)
        
        return Pose6D(x=x, y=y, z=z, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)


def check_for_end_or_abort(e):
    """Returns a closure checking for END or ABORT notifications."""
    def check(notification, e=e):
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return check

def calculate_pre_grasp_matrix(T_grasp_matrix: np.ndarray, distance: float) -> np.ndarray:
    """
    Calculates a pre-grasp 4x4 matrix by offsetting along the final pose's Z-axis.
    """
    approach_vector = T_grasp_matrix[:3, 2] # Z-axis is correct for the robot!
    T_pre_grasp = np.copy(T_grasp_matrix)
    T_pre_grasp[:3, 3] -= distance * approach_vector
    return T_pre_grasp
     

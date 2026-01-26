#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped, TransformStamped
from cv_bridge import CvBridge
import message_filters
import cv2
import threading
import numpy as np
import time
import math

from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_geometry_msgs

from image_geometry import PinholeCameraModel

from pymoveit2 import MoveIt2
from franka_msgs.action import Grasp, Move
from franka_msgs.msg import GraspEpsilon

from pipeline import OpenWorldGraspPipeline
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PARENT_FRAME = 'panda_hand'
CAMERA_FRAME = 'camera_color_optical_frame'

CAM_X = 0.06
CAM_Y = 0.025
CAM_Z = 0.05

CAM_QX = 0.0
CAM_QY = 0.0
CAM_QZ = 0.707
CAM_QW = 0.707

GRASP_EXTRA_DEPTH = 0.0        
GRIPPER_ROTATION_OFFSET = 0.0   
SAFE_HEIGHT = 0.04

def euler_to_quat(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [x, y, z, w]

class FrankaMove:
    def __init__(self, node, executor) -> None:
        self.node = node
        self.executor = executor
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
            base_link_name="panda_link0",
            end_effector_name="panda_hand_tcp",
            group_name="panda_arm",
            use_move_group_action=True,
        )
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1
        self.cli_gripper_move = ActionClient(self.node, Move, '/panda_gripper/move')
        self.cli_gripper_grasp = ActionClient(self.node, Grasp, '/panda_gripper/grasp')
        self.gripper_speed = 0.05
        self.gripper_width = 0.08
        self.gripper_force = 20.0

    def _spin_action_complete(self, goal_future):
        self.executor.spin_until_future_complete(goal_future)
        goal_handle = goal_future.result()
        if not goal_handle.accepted: return False
        result_future = goal_handle.get_result_async()
        self.executor.spin_until_future_complete(result_future)
        return result_future.result().result

    def _action_send(self, action, goal):
        goal_future = action.send_goal_async(goal)
        result = self._spin_action_complete(goal_future)
        return result

    def move_to_pose(self, position, quat_xyzw):
        self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw)
        return self._spin_action_complete(self.moveit2._MoveIt2__send_goal_future_move_action)

    def move_to_configuration(self, joint_positions):
        self.moveit2.move_to_configuration(joint_positions)
        return self._spin_action_complete(self.moveit2._MoveIt2__send_goal_future_move_action)

    def open(self):
        if not self.cli_gripper_move.server_is_ready(): return False
        g = Move.Goal(width=self.gripper_width, speed=self.gripper_speed)
        return self._action_send(self.cli_gripper_move, g)

    def close(self):
        if not self.cli_gripper_grasp.server_is_ready(): return False
        g = Grasp.Goal(
            width=0.0, speed=self.gripper_speed, force=self.gripper_force,
            epsilon=GraspEpsilon(inner=0.08, outer=0.08)
        )
        return self._action_send(self.cli_gripper_grasp, g)

class RosReader(Node):
    def __init__(self):
        super().__init__('owg_reader')
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_model = PinholeCameraModel()
        self.intrinsics_ready = False
        self.lock = threading.Lock()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_calibration()

        qos = qos_profile_sensor_data
        
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=qos)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.sync.registerCallback(self.sync_callback)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, qos)
        
        print("[READER] Initializing Pipeline...")
        self.pipeline = OpenWorldGraspPipeline()

    def publish_calibration(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = PARENT_FRAME
        t.child_frame_id = CAMERA_FRAME
        
        t.transform.translation.x = float(CAM_X)
        t.transform.translation.y = float(CAM_Y)
        t.transform.translation.z = float(CAM_Z)
        t.transform.rotation.x = float(CAM_QX)
        t.transform.rotation.y = float(CAM_QY)
        t.transform.rotation.z = float(CAM_QZ)
        t.transform.rotation.w = float(CAM_QW)

        self.tf_broadcaster.sendTransform(t)

    def info_callback(self, msg):
        if not self.intrinsics_ready:
            if hasattr(self.camera_model, 'from_camera_info'):
                self.camera_model.from_camera_info(msg)
            else:
                self.camera_model.fromCameraInfo(msg)
            self.intrinsics_ready = True
            print(f"[READER] Intrinsics Received: {msg.width}x{msg.height}")


    def sync_callback(self, rgb_msg, depth_msg):
        with self.lock:
            try:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            except Exception as e:
                print(f"[READER] Bridge Error: {e}")

    def get_latest_data(self):
        with self.lock:
            if self.latest_rgb is None or self.latest_depth is None or not self.intrinsics_ready:
                return None, None, None
            K = self.camera_model.intrinsic_matrix()
            return self.latest_rgb.copy(), self.latest_depth.copy(), K

    def transform_camera_to_robot(self, cam_x, cam_y, cam_z):
        try:
            pt_cam = PointStamped()
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.header.stamp = self.get_clock().now().to_msg()
            pt_cam.point.x = float(cam_x)
            pt_cam.point.y = float(cam_y)
            pt_cam.point.z = float(cam_z)

            target_frame = "panda_link0"
            current_time = self.get_clock().now()
            
            for attempt in range(3):
                if self.tf_buffer.can_transform(target_frame, CAMERA_FRAME, current_time, timeout=Duration(seconds=5)):
                    pt_robot = self.tf_buffer.transform(pt_cam, target_frame, timeout=Duration(seconds=5))
                    return [pt_robot.point.x, pt_robot.point.y, pt_robot.point.z]
                
                if attempt < 2:
                    time.sleep(0.5)
            
            print(f"[READER] Warn: Transform unavailable {CAMERA_FRAME} -> {target_frame}")
            return None

        except Exception as e:
            print(f"[READER] TF Error: {e}")
            return None

def main():
    logs_enabled = True
    for arg in sys.argv:
        if arg.lower() == 'logs:false':
            logs_enabled = False
        elif arg.lower() == 'logs:true':
            logs_enabled = True

    print(f"[READER] Logs enabled: {logs_enabled}")

    rclpy.init()
    
    reader_node = RosReader()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(reader_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    move_node = Node('franka_move_node')
    move_exec = rclpy.executors.SingleThreadedExecutor()
    move_exec.add_node(move_node)
    franka = FrankaMove(move_node, move_exec)
    
    print("[READER] Moving Home...")
    ready_state = [0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi]
    franka.move_to_configuration(ready_state)
    franka.open()

    print("[READER] Waiting for calibration...")
    time.sleep(2.0)
    print("[READER] Ready.")

    try:
        while rclpy.ok():
            print("\n[READER] Waiting for images...")
            rgb, depth, K = None, None, None
            while rgb is None:
                rgb, depth, K = reader_node.get_latest_data()
                if rgb is None: time.sleep(0.5)
            
            print("[READER] Captured.")
            user_query = input("Enter text query (or 'q' to quit): ").strip()
            if user_query.lower() == 'q': break
            if not user_query: continue

            print(f"[READER] Processing: '{user_query}'...")
            rgb, depth, K = reader_node.get_latest_data()

            result = reader_node.pipeline.run(
                image_input=rgb,
                depth_img=depth,
                intrinsics=K,
                user_query=user_query,
                logs=logs_enabled
            )

            if 'reasoning' in result and result['reasoning']:
                print(f"[READER] Model Thought: {result['reasoning']}")

            if 'error' in result:
                print(f"[READER] Error: {result['error']}")
            else:
                all_steps_complete = False
                current_steps = result.get('execution_steps', [])
                
                while not all_steps_complete:
                    if len(current_steps) == 0:
                        break
                    
                    step = current_steps[0]
                    
                    if 'error' in step:
                        print(f"[READER] Step Error: {step['error']}")
                        current_steps.pop(0)
                        continue
                    
                    action_type = step.get('action', 'unknown')
                    
                    if 'grasp_pose' in step and step['grasp_pose']:
                        cam_x = step['grasp_pose']['position'][0]
                        cam_y = step['grasp_pose']['position'][1]
                        cam_z = step['grasp_pose']['position'][2]
                        ang = step['grasp_pose']['angle']

                        print(f"[READER] Action: {action_type.upper()}")

                        robot_pos = reader_node.transform_camera_to_robot(cam_x, cam_y, cam_z)
                        
                        if robot_pos is None:
                            print("[READER] Transform failed.")
                            current_steps.pop(0)
                            continue

                        robot_x, robot_y, surface_z = robot_pos
                        
                        robot_z = surface_z - GRASP_EXTRA_DEPTH
                        if robot_z < SAFE_HEIGHT:
                            robot_z = SAFE_HEIGHT

                        target_pos = [robot_x, robot_y, robot_z]
                        
                        final_angle = -ang + GRIPPER_ROTATION_OFFSET
                        while final_angle > np.pi / 2: final_angle -= np.pi
                        while final_angle < -np.pi / 2: final_angle += np.pi

                        franka.open()
                        
                        target_quat = euler_to_quat(np.pi, 0, final_angle)
                        pre_grasp_pos = [robot_x, robot_y, robot_z + 0.15]
                        
                        print(f"[READER] Executing grasp sequence...")
                        franka.move_to_pose(pre_grasp_pos, target_quat)
                        time.sleep(0.2)
                        
                        franka.move_to_pose(target_pos, target_quat)
                        time.sleep(0.2)
                        
                        franka.close()
                        time.sleep(0.2)
                        
                        franka.move_to_pose(pre_grasp_pos, target_quat)
                        time.sleep(0.2)
                        
                        if action_type == "pick":
                            drop_config = [np.pi/2, -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi]
                        else:
                            drop_config = [-np.pi/2, -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi]
                        
                        franka.move_to_configuration(drop_config)
                        time.sleep(0.3)
                        
                        franka.open()
                        time.sleep(0.2)
                        
                        franka.move_to_configuration(ready_state)
                        time.sleep(0.3)
                        
                        current_steps.pop(0)
                        
                        if action_type == "remove":
                            print(f"[READER] Obstacle removed. Re-scanning...")
                            rgb_new, depth_new, K_new = None, None, None
                            
                            time.sleep(1.0)
                            
                            while rgb_new is None:
                                rgb_new, depth_new, K_new = reader_node.get_latest_data()
                                if rgb_new is None: time.sleep(0.5)
                            
                            result_new = reader_node.pipeline.run(
                                image_input=rgb_new,
                                depth_img=depth_new,
                                intrinsics=K_new,
                                user_query=user_query,
                                logs=logs_enabled
                            )
                            
                            if 'error' not in result_new:
                                current_steps = result_new.get('execution_steps', [])
                            else:
                                print(f"[READER] Pipeline Error: {result_new['error']}")
                                break
                        
                        if len(current_steps) == 0:
                            all_steps_complete = True
                    else:
                        current_steps.pop(0)
                        if len(current_steps) == 0:
                            all_steps_complete = True

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
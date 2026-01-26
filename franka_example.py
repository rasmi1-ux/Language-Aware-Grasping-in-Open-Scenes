#!/usr/bin/env python3
import numpy as np
import rclpy
import cv2
import os
import transforms3d.quaternions as tq
from rclpy.node import Node
from rclpy.action import ActionClient
from pymoveit2 import MoveIt2
from franka_msgs.action import Grasp, Move
from franka_msgs.msg import GraspEpsilon

from Grasp_detector import GraspDetector
from reader import FrankaDataReader
from segmentor import Segmentor

class FrankaMove:
    def __init__(self, node) -> None:
        self.node = node
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
        self.gripper_speed = float(0.05)
        self.gripper_width = float(0.08)
        self.gripper_force = float(1)

    def _spin_action_complete(self, goal_future):
        self.node.executor.spin_until_future_complete(goal_future)
        goal_handle = goal_future.result()
        result_future = goal_handle.get_result_async()
        self.node.executor.spin_until_future_complete(result_future)
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
        return self._action_send(self.cli_gripper_move, g).success

    def close(self):
        if not self.cli_gripper_grasp.server_is_ready(): return False
        g = Grasp.Goal(width=float(0), speed=self.gripper_speed, force=self.gripper_force, epsilon=GraspEpsilon(inner=np.inf, outer=np.inf))
        return self._action_send(self.cli_gripper_grasp, g).success

def deproject_pixel_to_point(u, v, z, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

def main():
    rclpy.init()
    node = Node("franka_move")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    franka = FrankaMove(node)

    base_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(base_path, 'data', '20240425_160916') 
    
    reader = FrankaDataReader(data_path)
    segmentor = Segmentor()
    detector = GraspDetector()

    print("Moving to Home...")
    ready_state = [0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi]
    franka.move_to_configuration(ready_state)
    franka.open()

    for i in range(len(reader)):
        rgb, depth, intrinsic, cam_pose = reader[i]
        seg_mask = segmentor.segment(rgb)
        
        obj_ids = np.unique(seg_mask)
        obj_ids = obj_ids[obj_ids != 0]

        for obj_id in obj_ids:
            current_obj_mask = (seg_mask == obj_id).astype(np.uint8)
            rows = np.any(current_obj_mask, axis=1)
            cols = np.any(current_obj_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols): continue
                 
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            candidates = detector.detect_candidates(rgb, depth, current_obj_mask, (cmin, rmin, cmax, rmax), intrinsic)
            
            if candidates:
                best_grasp = candidates[0] 
                u, v = best_grasp['uv']
                z_depth = best_grasp['z']
                angle = best_grasp['angle']

                point_cam = deproject_pixel_to_point(u, v, z_depth, intrinsic)
                point_cam_hom = np.append(point_cam, 1.0)
                point_world = (cam_pose @ point_cam_hom)[:3]
                
                print(f"Executing Grasp at: {point_world}")
                
                q_grasp = tq.axangle2quat([0, 0, 1], angle)
                q_align = [0, 1, 0, 0] 
                
                franka.move_to_pose(position=point_world.tolist(), quat_xyzw=[1, 0, 0, 0])
                franka.close()
                franka.move_to_configuration(ready_state)
                franka.open()
                break 

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
import os
import re
import cv2
import json
import time
import psutil
import numpy as np
import torch
from segmentor import SAM2Segmenter
from model import QwenOWG
from Grasp_detector import GraspDetector

class PerformanceTracker:
    def __init__(self):
        self.stages = {}
        self.process = psutil.Process(os.getpid())
    
    def start_stage(self, stage_name):
        self.stages[stage_name] = {
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / (1024 ** 2)
        }
    
    def end_stage(self, stage_name):
        if stage_name in self.stages:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / (1024 ** 2)
            
            self.stages[stage_name]['elapsed_time'] = end_time - self.stages[stage_name]['start_time']
            self.stages[stage_name]['memory_used'] = end_memory - self.stages[stage_name]['start_memory']
            self.stages[stage_name]['peak_memory'] = end_memory
    
    def print_report(self):
        pass

class OpenWorldGraspPipeline:
    def __init__(self, 
                 sam_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
                 sam_checkpoint="sam2.1_hiera_base_plus.pt",
                 qwen_path="Qwen"):
        
        torch.set_num_threads(4)
        self.perf_tracker = PerformanceTracker()
        
        print("\n[PIPELINE] Initializing SAM2...")
        self.sam = SAM2Segmenter(config_path=sam_config, checkpoint_path=sam_checkpoint)

        print("[PIPELINE] Initializing QwenOWG...")
        self.qwen = QwenOWG(qwen_path)
        
        print(f"[PIPELINE] Initializing Grasp Detector...")
        self.grasp_detector = GraspDetector()
        print("[PIPELINE] Ready.\n")

    def _back_project(self, u, v, z, intrinsics):
        if isinstance(intrinsics, dict): K = intrinsics['K']
        else: K = intrinsics
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return [x, y, z]

    def _get_robust_z(self, depth_img, u, v, kernel=5):
        if depth_img is None: return 0.5
        h, w = depth_img.shape
        u, v = int(u), int(v)
        
        v_min, v_max = max(0, v - kernel), min(h, v + kernel)
        u_min, u_max = max(0, u - kernel), min(w, u + kernel)
        
        patch = depth_img[v_min:v_max, u_min:u_max]
        valid = patch[patch > 0]
        
        if valid.size == 0: return 0.5
        
        val = np.median(valid)
        if depth_img.dtype == np.uint16:
            val /= 1000.0
        return float(val)

    def run(self, image_input, user_query, depth_img=None, intrinsics=None, logs=True):
        self.perf_tracker = PerformanceTracker()
        
        if isinstance(image_input, np.ndarray):
            if logs:
                temp_path = "temp_live_capture.png"
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_path, image_bgr)
                input_path = temp_path
            else:
                input_path = image_input
        else:
            input_path = image_input

        print(f"[PIPELINE] Segmenting & Grounding: '{user_query}'")
        
        self.perf_tracker.start_stage("Segmentation (SAM2)")
        raw_path, seg_path, instance_mask, masks = self.sam.segment(input_path, save_to_disk=logs)
        self.perf_tracker.end_stage("Segmentation (SAM2)")
        
        self.perf_tracker.start_stage("Foreground (Qwen)")
        ground_out = self.qwen.F_ground(raw_path, seg_path, user_query)
        self.perf_tracker.end_stage("Foreground (Qwen)")
        
        match = re.search(r"\[(.*?)\]", ground_out)
        target_id = None
        if match:
            try:
                id_list = [int(x.strip()) for x in match.group(1).split(",")]
                target_id = id_list[0]
            except ValueError:
                pass
        
        if target_id is None:
            return {"error": "Target ID extraction failed", "ground_raw": ground_out}

        self.perf_tracker.start_stage("Planning (Qwen)")
        plan_out = self.qwen.F_plan(seg_path, target_id, raw_img=image_input)
        self.perf_tracker.end_stage("Planning (Qwen)")
        
        execution_steps = []
        reasoning = plan_out
        
        try:
            json_match = re.search(r"\[.*\]", plan_out, re.DOTALL)
            if json_match:
                plan_actions = json.loads(json_match.group(0))
                reasoning = plan_out[:json_match.start()].strip()
            else:
                plan_actions = [{"action": "pick", "input": target_id}]
        except:
            plan_actions = [{"action": "pick", "input": target_id}]

        last_vis_path = None 
        
        if depth_img is not None and intrinsics is not None:
            if plan_actions and len(plan_actions) > 0:
                step = plan_actions[0]
                step_result = step.copy()
                obj_id = step.get("input")
                action = step.get("action")
                
                obj_mask = (instance_mask == obj_id).astype(np.uint8)
                if np.count_nonzero(obj_mask) == 0:
                    step_result["error"] = "Mask empty"
                    execution_steps.append(step_result)
                    return {
                        "execution_steps": execution_steps,
                        "last_vis_path": last_vis_path,
                        "reasoning": reasoning
                    }

                if action == "pick" or action == "remove":
                    print(f"[PIPELINE] Performing {action.upper()} on object {obj_id}")
                    
                    if logs:
                        rgb_cv = cv2.imread(raw_path)
                    else:
                        if isinstance(raw_path, str):
                            rgb_cv = cv2.imread(raw_path)
                        else:
                            rgb_cv = cv2.cvtColor(np.array(raw_path), cv2.COLOR_RGB2BGR)
                    
                    ys, xs = np.where(obj_mask > 0)
                    y_min, y_max = ys.min(), ys.max()
                    x_min, x_max = xs.min(), xs.max()
                    
                    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
                    size = max(y_max - y_min, x_max - x_min)
                    crop_size = int(size * 2.0)
                    half = crop_size // 2
                    
                    h_img, w_img = rgb_cv.shape[:2]
                    y1, y2 = max(0, cy - half), min(h_img, cy + half)
                    x1, x2 = max(0, cx - half), min(w_img, cx + half)
                    
                    cropped_rgb = rgb_cv[y1:y2, x1:x2]
                    cropped_depth = depth_img[y1:y2, x1:x2]
                    cropped_mask = obj_mask[y1:y2, x1:x2]
                    h_crop_real, w_crop_real = cropped_rgb.shape[:2]

                    NET_SIZE = 224
                    rgb_res = cv2.resize(cropped_rgb, (NET_SIZE, NET_SIZE))
                    depth_res = cv2.resize(cropped_depth, (NET_SIZE, NET_SIZE))
                    mask_res = cv2.resize(cropped_mask, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_NEAREST)

                    bin_mask = (mask_res > 0).astype(np.float32)
                    rgb_res = (rgb_res.astype(np.float32) * np.stack([bin_mask]*3, axis=-1)).astype(np.uint8)
                    depth_res = (depth_res * bin_mask).astype(depth_res.dtype)

                    self.perf_tracker.start_stage("Grasping (Detection)")
                    candidates = self.grasp_detector.detect_candidates(
                        rgb_res, depth_res, mask_res, (0, 0, NET_SIZE, NET_SIZE), intrinsics
                    )
                    self.perf_tracker.end_stage("Grasping (Detection)")
                    
                    valid_candidates = []
                    scale_x = w_crop_real / float(NET_SIZE)
                    scale_y = h_crop_real / float(NET_SIZE)

                    for cand in candidates:
                        u_net, v_net = cand['uv']
                        cand['uv'] = [u_net * scale_x + x1, v_net * scale_y + y1]
                        cand['width'] = cand['width'] * scale_x
                        
                        u_g, v_g = int(cand['uv'][0]), int(cand['uv'][1])
                        if 0 <= v_g < h_img and 0 <= u_g < w_img:
                             if obj_mask[v_g, u_g] > 0:
                                valid_candidates.append(cand)

                    candidates = valid_candidates

                    if candidates:
                        all_grasps_path, _ = self.grasp_detector.create_single_image_with_all_grasps(rgb_cv, candidates, obj_mask, save_to_disk=logs)
                        
                        self.perf_tracker.start_stage("Grasp Selection (Qwen)")
                        rank_out = self.qwen.F_rank(all_grasps_path)
                        self.perf_tracker.end_stage("Grasp Selection (Qwen)")
                        
                        best_idx = 0 
                        match = re.search(r"\[(.*?)\]", rank_out)
                        if match:
                             try:
                                 ids = [int(x.strip()) for x in match.group(1).split(",")]
                                 if ids and ids[0] <= len(candidates): best_idx = ids[0] - 1
                             except: pass
                        
                        final_vis_path, _ = self.grasp_detector.create_single_image_with_all_grasps(
                            rgb_cv, candidates, obj_mask, best_idx=best_idx, save_to_disk=logs
                        )
                        
                        best_grasp = candidates[best_idx]
                        u, v = best_grasp['uv']
                        angle = best_grasp['angle']

                        if action == "remove":
                            corrected_angle = self.grasp_detector._correct_grasp_angle(obj_mask, angle)
                            angle = corrected_angle
                        
                        z = self._get_robust_z(depth_img, u, v)
                        x_m, y_m, z_m = self._back_project(u, v, z, intrinsics)
                        
                        width_px = max(best_grasp['width'] * 2.0, 30.0)
                        
                        step_result["grasp_pose"] = {
                            "position": [x_m, y_m, z_m],
                            "angle": angle,
                            "width": width_px 
                        }
                        
                        step_result["vis_path"] = final_vis_path
                        last_vis_path = final_vis_path
                    else:
                        step_result["error"] = f"No valid grasps found for {action}"
                        print(f"[PIPELINE] ERROR: No valid grasp candidates found for {obj_id}")
                else:
                    step_result["error"] = f"Unknown action: {action}"
                    print(f"[PIPELINE] ERROR: Unknown action type: {action}")
                
                execution_steps.append(step_result)
            else:
                print(f"[PIPELINE] ERROR: No actions to process")
        else:
            print(f"[PIPELINE] WARNING: Missing depth_img or intrinsics.")

        return {
            "execution_steps": execution_steps,
            "last_vis_path": last_vis_path,
            "reasoning": reasoning
        }
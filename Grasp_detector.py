import os
import cv2
import numpy as np
import torch
from PIL import Image

class GraspDetector:
    def __init__(self):
        torch.set_num_threads(4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _generate_geometry_based_candidates(self, mask, depth, intrinsics):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return []
        
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return []
        
        rect = cv2.minAreaRect(contour)
        center, (width, height), angle = rect
        center_x, center_y = int(center[0]), int(center[1])
        
        dominant_angle = np.deg2rad(angle)
        
        if width < height:
            width, height = height, width
            dominant_angle += np.pi / 2
        
        grasp_angle = dominant_angle + np.pi / 2
        
        candidates = []
        
        num_grasps = 3
        dx = np.cos(dominant_angle)
        dy = np.sin(dominant_angle)
        
        for i in range(num_grasps):
            offset = (i - num_grasps // 2) * width / (num_grasps + 1)
            
            u = center_x + offset * dx
            v = center_y + offset * dy
            
            u, v = int(u), int(v)
            
            h, w = mask.shape
            if 0 <= u < w and 0 <= v < h and mask[v, u] > 0:
                z = 0.5
                if 0 <= v < h and 0 <= u < w:
                    v_min, v_max = max(0, v-2), min(h, v+3)
                    u_min, u_max = max(0, u-2), min(w, u+3)
                    patch = depth[v_min:v_max, u_min:u_max]
                    valid = patch[patch > 0]
                    if valid.size > 0:
                        z = float(np.median(valid))
                
                grasp_width = height * 0.8
                
                candidates.append({
                    'uv': (u, v),
                    'angle': grasp_angle,
                    'width': grasp_width,
                    'quality': 0.8,
                    'z': float(z)
                })
        
        return candidates

    def _get_object_dominant_angle(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        if len(contour) < 5:
            return None
        
        points = contour.reshape(-1, 2).astype(np.float32)
        
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        cov = np.cov(centered.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        dominant_idx = np.argmax(eigenvalues)
        dominant_vector = eigenvectors[:, dominant_idx]
        
        dominant_angle = np.arctan2(dominant_vector[1], dominant_vector[0])
        
        if dominant_angle < 0:
            dominant_angle += np.pi
        if dominant_angle > np.pi:
            dominant_angle -= np.pi
        
        return dominant_angle

    def _correct_grasp_angle(self, mask, predicted_angle):
        dominant_angle = self._get_object_dominant_angle(mask)
        
        if dominant_angle is None:
            return predicted_angle
        
        norm_pred = predicted_angle % np.pi
        if norm_pred < 0:
            norm_pred += np.pi
        
        norm_dom = dominant_angle % np.pi
        if norm_dom < 0:
            norm_dom += np.pi
        
        perp1 = norm_dom
        perp2 = (norm_dom + np.pi / 2) % np.pi
        
        dist_to_perp1 = min(abs(norm_pred - perp1), np.pi - abs(norm_pred - perp1))
        dist_to_perp2 = min(abs(norm_pred - perp2), np.pi - abs(norm_pred - perp2))
        
        if dist_to_perp1 <= dist_to_perp2:
            corrected = perp1
        else:
            corrected = perp2
        
        return corrected

    def detect_candidates(self, rgb, depth, mask, box, intrinsics):
        x1, y1, x2, y2 = box
        h, w = rgb.shape[:2]
        pad = 40
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(w, x2+pad); y2 = min(h, y2+pad)

        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        if rgb_crop.size == 0: return []

        geometry_candidates = self._generate_geometry_based_candidates(mask_crop, depth_crop, intrinsics)
        
        for cand in geometry_candidates:
            u_net, v_net = cand['uv']
            cand['uv'] = [u_net + x1, v_net + y1]
        
        return geometry_candidates

    def create_single_image_with_all_grasps(self, rgb_img, candidates, target_mask=None, best_idx=None, save_to_disk=True):
        if not candidates:
            return None, []
        
        vis_img = rgb_img.copy()
        
        if target_mask is not None:
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
        
        if target_mask is not None:
            ys, xs = np.where(target_mask > 0)
            if len(ys) > 0:
                obj_height = ys.max() - ys.min()
                obj_width = xs.max() - xs.min()
                uniform_width = max(obj_height, obj_width) * 0.4
            else:
                uniform_width = 80
        else:
            uniform_width = 80
        
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (255, 128, 0),
        ]
        
        valid_indices = []
        
        for i, cand in enumerate(candidates):
            cand_id = i + 1
            valid_indices.append(cand_id)
            
            color = colors[i % len(colors)]
            
            c_x, c_y = cand['uv']
            theta = cand['angle']
            
            width = uniform_width
            
            dx = (width / 2) * np.cos(theta)
            dy = (width / 2) * np.sin(theta)
            p1 = (int(c_x - dx), int(c_y - dy))
            p2 = (int(c_x + dx), int(c_y + dy))
            
            plate_h = width * 0.3
            p_dx = plate_h * np.cos(theta + np.pi/2)
            p_dy = plate_h * np.sin(theta + np.pi/2)
            
            line_thickness = 4 if (best_idx is not None and i == best_idx) else 2
            
            cv2.line(vis_img, (int(p1[0]-p_dx), int(p1[1]-p_dy)), 
                     (int(p1[0]+p_dx), int(p1[1]+p_dy)), color, line_thickness)
            cv2.line(vis_img, (int(p2[0]-p_dx), int(p2[1]-p_dy)), 
                     (int(p2[0]+p_dx), int(p2[1]+p_dy)), color, line_thickness)
            cv2.line(vis_img, p1, p2, color, line_thickness)
            
            circle_radius = 8 if (best_idx is not None and i == best_idx) else 5
            cv2.circle(vis_img, (int(c_x), int(c_y)), circle_radius, color, -1)
            cv2.putText(vis_img, f"G{cand_id}", (int(c_x) + 10, int(c_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if best_idx is not None and i == best_idx:
                cv2.circle(vis_img, (int(c_x), int(c_y)), circle_radius + 8, (0, 255, 0), 3)
                cv2.putText(vis_img, "BEST", (int(c_x) - 20, int(c_y) + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_to_disk:
            final_vis_path = "temp_all_grasps_single_image.png"
            cv2.imwrite(final_vis_path, vis_img)
            return final_vis_path, valid_indices
        else:
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(vis_img_rgb), valid_indices
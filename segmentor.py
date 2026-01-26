import os
import uuid
import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sam2")
torch.set_num_threads(4)

class SAM2Segmenter:
    def __init__(self, config_path, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = build_sam2(config_path, checkpoint_path, device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model,
            pred_iou_thresh=0.92,
            stability_score_thresh=0.92,
        )

    @staticmethod
    def _random_color(seed):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=3, dtype=np.uint8)

    @staticmethod
    def _find_best_id_position(mask, cx, cy, masks, obj_id):
        ys, xs = np.where(mask)
        if xs.size == 0:
            return cx, cy
        
        obj_size = np.sum(mask)
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        
        for other_id, other_mask in enumerate(masks):
            if other_id == obj_id - 1:
                continue
            
            other_seg = other_mask["segmentation"]
            overlap = mask & other_seg
            if not np.any(overlap):
                continue
            
            other_size = np.sum(other_seg)
            
            if obj_size < other_size:
                return cx, cy
            else:
                other_ys, other_xs = np.where(other_seg)
                width = max_x - min_x
                height = max_y - min_y
                
                candidates = [
                    (min_x + width * 0.25, min_y + height * 0.25),
                    (max_x - width * 0.25, min_y + height * 0.25),
                    (min_x + width * 0.25, max_y - height * 0.25),
                    (max_x - width * 0.25, max_y - height * 0.25),
                    (min_x + width * 0.2, cy),
                    (max_x - width * 0.2, cy),
                    (cx, min_y + height * 0.2),
                    (cx, max_y - height * 0.2),
                ]
                
                best_candidate = cx, cy
                best_distance = 0
                
                for cand_x, cand_y in candidates:
                    int_x, int_y = int(cand_x), int(cand_y)
                    if 0 <= int_y < mask.shape[0] and 0 <= int_x < mask.shape[1]:
                        if mask[int_y, int_x]:
                            overlap_ys, overlap_xs = np.where(overlap)
                            if overlap_xs.size > 0:
                                overlap_cx = overlap_xs.mean()
                                overlap_cy = overlap_ys.mean()
                                dist = np.sqrt((cand_x - overlap_cx)**2 + (cand_y - overlap_cy)**2)
                                if dist > best_distance:
                                    best_distance = dist
                                    best_candidate = (cand_x, cand_y)
                
                return best_candidate
        
        return cx, cy

    def _filter_overlapping_masks(self, masks, iou_thresh=0.7, containment_thresh=0.8):
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        kept_masks = []
        
        for mask_data in masks:
            mask = mask_data["segmentation"]
            mask_area = mask_data["area"]
            
            should_keep = True
            for kept_data in kept_masks:
                kept_mask = kept_data["segmentation"]
                
                intersection = np.logical_and(mask, kept_mask).sum()
                if intersection == 0:
                    continue
                
                union = np.logical_or(mask, kept_mask).sum()
                iou = intersection / union
                containment = intersection / mask_area
                
                if iou > iou_thresh or containment > containment_thresh:
                    should_keep = False
                    break
            
            if should_keep:
                kept_masks.append(mask_data)
        
        return kept_masks

    def _add_ids_to_image(self, image, masks):
        image_with_ids = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)
        
        for obj_id, m in enumerate(masks, start=1):
            mask = m["segmentation"]
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            
            cx, cy = xs.mean(), ys.mean()
            best_x, best_y = self._find_best_id_position(mask, cx, cy, masks, obj_id)
            
            text = str(obj_id)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            margin = 5
            cv2.rectangle(
                image_with_ids,
                (int(best_x - text_size[0]//2 - margin), int(best_y - text_size[1]//2 - margin)),
                (int(best_x + text_size[0]//2 + margin), int(best_y + text_size[1]//2 + margin)),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                image_with_ids,
                text,
                (int(best_x - text_size[0]//2), int(best_y + text_size[1]//2)),
                font,
                font_scale,
                text_color,
                font_thickness
            )
        
        return image_with_ids

    def segment(self, image_input, save_to_disk=True):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            image_np = np.array(image)
        else:
            image_np = image_input
            image = Image.fromarray(image_np)

        masks = self.mask_generator.generate(image_np)
        masks = self._filter_overlapping_masks(masks)

        h, w = image_np.shape[:2]

        instance_mask = np.zeros((h, w), dtype=np.uint16)
        overlay = image_np.copy()

        for obj_id, m in enumerate(masks, start=1):
            mask = m["segmentation"]
            instance_mask[mask] = obj_id
            color = self._random_color(obj_id)
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)

        raw_with_ids = self._add_ids_to_image(image_np, masks)
        overlay_with_ids = self._add_ids_to_image(overlay, masks)

        if save_to_disk:
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            raw_with_ids_bgr = cv2.cvtColor(raw_with_ids, cv2.COLOR_RGB2BGR)
            overlay_with_ids_bgr = cv2.cvtColor(overlay_with_ids, cv2.COLOR_RGB2BGR)

            run_id = str(uuid.uuid4())[:8]
            out_dir = f"runs/{run_id}"
            os.makedirs(out_dir, exist_ok=True)

            raw_path = f"{out_dir}/raw.png"
            raw_ids_path = f"{out_dir}/raw_with_ids.png"
            seg_path = f"{out_dir}/segmented.png"

            cv2.imwrite(raw_path, image_np_bgr)
            cv2.imwrite(raw_ids_path, raw_with_ids_bgr)
            cv2.imwrite(seg_path, overlay_with_ids_bgr)

            np.save(f"{out_dir}/instance_mask.npy", instance_mask)
            np.save(f"{out_dir}/masks.npy", masks)

            return raw_path, seg_path, instance_mask, masks
        else:
            raw_ids_pil = Image.fromarray(raw_with_ids)
            seg_pil = Image.fromarray(overlay_with_ids)
            return raw_ids_pil, seg_pil, instance_mask, masks
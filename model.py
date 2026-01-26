import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class QwenOWG:
    def __init__(self, model_path, prompts_dir="prompts"):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.f_ground_prompt = open(f"{prompts_dir}/referring_segmentation.txt").read()
        self.f_plan_prompt   = open(f"{prompts_dir}/grasp_planning.txt").read()
        self.f_rank_prompt   = open(f"{prompts_dir}/grasp_ranking.txt").read()

    def _run(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        trimmed = output_ids[:, inputs.input_ids.shape[1]:]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def F_ground(self, raw_img, seg_img, user_query):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": self.f_ground_prompt},
                    {"type": "image", "image": raw_img},
                    {"type": "image", "image": seg_img},
                    {"type": "text",  "text": user_query},
                ]
            }
        ]
        return self._run(messages)

    def F_plan(self, seg_img, target_id, raw_img=None):
        plan_query = f"Target object ID: {target_id}"
        content = [
            {"type": "text",  "text": self.f_plan_prompt},
        ]
        
        if raw_img is not None:
            content.append({"type": "image", "image": raw_img})
        
        content.append({"type": "image", "image": seg_img})
        content.append({"type": "text",  "text": plan_query})
        
        messages = [{"role": "user", "content": content}]
        return self._run(messages)

    def F_rank(self, grasp_img):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": self.f_rank_prompt},
                    {"type": "image", "image": grasp_img},
                ]
            }
        ]
        return self._run(messages)

import json
import os
import torch
from PIL import Image
from config import DATASET_PATH, RESULTS_PATH, PROMPTS, REPROMPT_TIMES

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Enable TF32 matmul for faster GPU ops where supported
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_name):
    # Optionally use Flash Attention 2 for speed and lower memory: set attn_implementation="flash_attention_2"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,  # bfloat16 is recommended
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Important: left padding is required for batched generation
    processor.tokenizer.padding_side = "left"

    model.eval()
    print("Model and processor loaded.")
    return model, processor


def run_inference_batch(model, processor, image_objs, prompts, max_new_tokens=32):
    """
    Manual batched inference:
    1. Process each sample independently (avoids feature/token mismatches)
    2. Apply left padding and concatenate features into one batch
    """
    processed_samples = []

    # --- 1. Per-sample processing ---
    for img, p in zip(image_objs, prompts):
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                    "max_pixels": 1024 * 1024,
                },
                {"type": "text", "text": p}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        inputs = {k: v.cpu() for k, v in inputs.items()}
        processed_samples.append(inputs)

    # --- 2. Collate into a batch ---
    batch_size = len(processed_samples)

    input_ids_list = [s["input_ids"][0] for s in processed_samples]
    max_len = max(len(ids) for ids in input_ids_list)
    pad_token_id = processor.tokenizer.pad_token_id

    batched_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    batched_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, ids in enumerate(input_ids_list):
        seq_len = len(ids)
        batched_input_ids[i, -seq_len:] = ids
        batched_attention_mask[i, -seq_len:] = 1

    batched_inputs = {
        "input_ids": batched_input_ids.to(model.device),
        "attention_mask": batched_attention_mask.to(model.device)
    }

    keys_to_merge = [k for k in processed_samples[0].keys() if k not in ["input_ids", "attention_mask"]]
    for k in keys_to_merge:
        batched_inputs[k] = torch.cat([s[k] for s in processed_samples], dim=0).to(model.device)

    # --- 3. Inference ---
    with torch.inference_mode():
        generated_ids = model.generate(
            **batched_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            # For stochastic reprompt runs with variation, use e.g.:
            # do_sample=True, temperature=0.7, top_p=0.9
        )

    # --- 4. Decode (trim prompt tokens) ---
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_list, generated_ids)
    # ]
    input_token_len = batched_inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_len:]

    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)


def check_correctness(response, ground_truth):
    clean_resp = response.lower().strip().replace("_", " ").replace("-", " ")
    clean_gt = ground_truth.lower().strip().replace("_", " ").replace("-", " ")
    return clean_gt in clean_resp


def run_evaluation(model_name, batch_size=16):
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    model, processor = load_model(model_name)

    results_file = RESULTS_PATH.format(model=model_name)
    if os.path.exists(results_file):
        os.remove(results_file)

    wf = open(results_file, "a", encoding="utf-8", buffering=1 << 20)

    def dump(entry):
        wf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(dataset)
    idx = 0
    print(f"Starting evaluation on {total} items...")

    while idx < total:
        batch_items = dataset[idx: idx + batch_size]

        imgs, metas, recog_prompts = [], [], []

        # --- 1. Data Loading ---
        for it in batch_items:
            image_path = it["image_path"]

            meta = {
                "image_id": it.get("id"),
                "breed": it["breed"],
                "animal_type": it["animal_type"],
                "font_category": it.get("font_category"),
                "font_name": it.get("font_name"),
                "size": it.get("size"),
                "position_id": it.get("position_id"),
                "text_color_name": it.get("text_color_name"),
            }

            if not os.path.exists(image_path):
                dump({**meta, "error": "image_not_found", "is_correct": False})
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                imgs.append(img)
                metas.append(meta)
                recog_prompts.append(
                    PROMPTS["recognition"].format(super_category=it["animal_type"])
                )
            except Exception as e:
                dump({**meta, "error": f"preprocess_error: {e}", "is_correct": False})
                continue

        # --- 2. Inference ---
        if imgs:
            try:
                recog_outputs = run_inference_batch(
                    model, processor, imgs, recog_prompts, max_new_tokens=24
                )
            except Exception as e:
                print(f"!!! Batch Failed at idx {idx}: {e}")
                for meta in metas:
                    dump({**meta, "error": f"batch_error: {repr(e)}", "is_correct": False})
                idx += batch_size
                continue

            # --- 3. Save Results ---
            for meta, rec in zip(metas, recog_outputs):
                is_correct = check_correctness(rec, meta["breed"])

                dump({
                    **meta,
                    "recog_output": rec,
                    "is_correct": is_correct
                })

        idx += batch_size
        if idx % (batch_size * 5) == 0:
            print(f"Processed {idx}/{total}")
            wf.flush()

    wf.close()
    print("Evaluation Complete.")


if __name__ == "__main__":
    run_evaluation(model_name="qwen", batch_size=16)

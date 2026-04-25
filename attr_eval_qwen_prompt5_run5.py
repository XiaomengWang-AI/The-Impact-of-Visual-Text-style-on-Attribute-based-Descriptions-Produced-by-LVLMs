import json
import os
import torch
from PIL import Image
from config import DATASET_PATH, RESULTS_PATH, PROMPTS, REPROMPT_TIMES

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Enable TF32 acceleration
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_name):
    # Optionally use Flash Attention 2 for speed and lower VRAM (uncomment attn_implementation).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,  # Prefer bfloat16
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Important: left padding is required for batch inference.
    processor.tokenizer.padding_side = "left"

    model.eval()
    print("Model and processor loaded.")
    return model, processor


def run_inference_batch(model, processor, image_objs, prompts, max_new_tokens=32):
    """
    Manual batching:
    1. Process each sample separately (avoids feature/token mismatches).
    2. Manually apply left padding and concatenate features.
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

    # --- 2. Manual batch collate ---
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
            # For different outputs across reprompt runs, use e.g.:
            # do_sample=True, temperature=0.7, top_p=0.9
        )

    # --- 4. Decode (trim to new tokens) ---
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


def run_evaluation(model_name, batch_size=16):  # Reduce batch_size if VRAM is tight
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    model, processor = load_model(model_name)

    results_file = RESULTS_PATH.format(model=model_name)
    if os.path.exists(results_file):
        os.remove(results_file)

    wf = open(results_file, "a", encoding="utf-8", buffering=1 << 20)

    def dump(entry):
        wf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Prompts to run: all PROMPTS keys except "recognition".
    prompt_keys = [k for k in PROMPTS.keys() if k != "recognition"]
    if len(prompt_keys) == 0:
        print("WARNING: No prompts to traverse (PROMPTS has only 'recognition'?).")

    total = len(dataset)
    idx = 0
    print(f"Starting evaluation on {total} items...")

    while idx < total:
        batch_items = dataset[idx: idx + batch_size]

        imgs, metas, recog_prompts = [], [], []

        # 1) Data Loading
        for it in batch_items:
            image_path = it["image_path"]
            meta = {
                "image_id": it.get("id"),
                "breed": it["breed"],
                "animal_type": it["animal_type"],
            }
            if not os.path.exists(image_path):
                dump({**meta, "error": "image_not_found", "is_correct": False})
                continue
            try:
                img = Image.open(image_path).convert("RGB")
                imgs.append(img)
                metas.append(meta)
                recog_prompts.append(PROMPTS["recognition"].format(super_category=it["animal_type"]))
            except Exception as e:
                dump({**meta, "error": f"preprocess_error: {e}", "is_correct": False})
                continue

        # 2) Inference (recognition)
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

            # --- Step 2: keep only correctly recognized samples ---
            correct_imgs = []
            correct_metas = []
            correct_recog_results = []

            for meta, img, rec in zip(metas, imgs, recog_outputs):
                is_correct = check_correctness(rec, meta["breed"])
                if not is_correct:
                    dump({
                        **meta,
                        "recog_output": rec,
                        "is_correct": False,
                        "attr_output": None
                    })
                else:
                    correct_imgs.append(img)
                    correct_metas.append(meta)
                    correct_recog_results.append(rec)

            # --- Step 3: for each correct image, run attribute prompts REPROMPT_TIMES ---
            if correct_imgs and prompt_keys:
                # Init structure: attr_output[prompt_key] = {"runs": [...]}
                per_image_attr = [
                    {pk: {"runs": []} for pk in prompt_keys}
                    for _ in range(len(correct_imgs))
                ]

                for r in range(REPROMPT_TIMES):
                    flat_imgs = []
                    flat_prompts = []
                    flat_map = []  # (img_index, prompt_key)

                    for i, meta in enumerate(correct_metas):
                        for pk in prompt_keys:
                            tpl = PROMPTS[pk]
                            try:
                                prompt = tpl.format(super_category=meta["animal_type"])
                            except Exception:
                                prompt = tpl

                            flat_imgs.append(correct_imgs[i])
                            flat_prompts.append(prompt)
                            flat_map.append((i, pk))

                    try:
                        flat_outputs = run_inference_batch(
                            model, processor, flat_imgs, flat_prompts, max_new_tokens=48
                        )
                    except Exception as e:
                        print(f"!!! Attribute Batch Failed (reprompt {r+1}/{REPROMPT_TIMES}): {e}")
                        flat_outputs = [f"Error: {str(e)}"] * len(flat_imgs)

                    # Map outputs back and append to runs
                    for (img_i, pk), out in zip(flat_map, flat_outputs):
                        per_image_attr[img_i][pk]["runs"].append(out)

                # Write one JSON line per correct image with final fields
                for meta, rec, attr_dict in zip(correct_metas, correct_recog_results, per_image_attr):
                    dump({
                        **meta,
                        "recog_output": rec,
                        "is_correct": True,
                        "attr_output": attr_dict
                    })

            # If recognition succeeded but there are no attribute prompts, still write rows
            elif correct_imgs and not prompt_keys:
                for meta, rec in zip(correct_metas, correct_recog_results):
                    dump({
                        **meta,
                        "recog_output": rec,
                        "is_correct": True,
                        "attr_output": {}
                    })

        idx += batch_size
        if idx % (batch_size * 5) == 0:
            print(f"Processed {idx}/{total}")
            wf.flush()

    wf.close()
    print("Evaluation Complete.")


if __name__ == "__main__":
    run_evaluation(model_name="qwen", batch_size=16)

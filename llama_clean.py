import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig



def normalize_runs_payload(payload: Any) -> List[str]:
    if payload is None:
        return [""]

    if isinstance(payload, dict) and "runs" in payload:
        runs = payload["runs"]
        if isinstance(runs, list):
            return ["" if x is None else str(x) for x in runs]
        return [str(runs)]

    if isinstance(payload, list):
        return ["" if x is None else str(x) for x in payload]

    return [str(payload)]


def clean_one_text(raw: str) -> str:
    s = raw.strip()

    # 去 assistant 噪声
    s = re.sub(r"^\s*assistant\s*", "", s, flags=re.IGNORECASE)

    # 👉 新增：统一分隔符
    # s = s.replace(": ", ",")
    # s = s.replace(";", ",")
    # s = s.replace("|", ",")

    return s.strip()


# ----------------- LLaMA -----------------

SYSTEM_PROMPT_DEFAULT = (
  `  "Act as a text processing system. "
    "Extract adjectives (or adjective phrases) from the input and output only a single-line list separated by comma. "
    "No other text."`
)


def build_llama_pipe(model_id: str, load_4bit=True):
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )


def run_llama(text, pipe, system_prompt, max_new_tokens):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    generated = outputs[0]["generated_text"]

    if isinstance(generated, list):
        return generated[-1]["content"].strip()

    return str(generated).strip()


# ----------------- JSON IO -----------------

def load_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        first = ""
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                first = line.lstrip()
                f.seek(pos)
                break

        if first[0] in ["[", "{"]:
            try:
                return ("json", json.load(f))
            except:
                f.seek(0)

        f.seek(0)
        return ("jsonl", [json.loads(l) for l in f if l.strip()])


def save_json_any(path, fmt, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if fmt == "json":
            json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            for r in obj:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--field_in", default="attr_output")
    ap.add_argument("--field_out", default="attr_output_clean")
    args = ap.parse_args()

    print("Loading JSON...")
    fmt, data = load_json_any(args.in_path)
    records = data if fmt == "jsonl" else data

    print("Loading LLaMA...")
    pipe = build_llama_pipe(args.model)

    for rec in tqdm(records, desc="Cleaning attr_output.runs"):
        attr = rec.get(args.field_in)

        if not isinstance(attr, dict):
            rec[args.field_out] = None
            continue

        attr_clean = {}

        for prompt_key, payload in attr.items():
            runs = normalize_runs_payload(payload)

            cleaned_runs = []

            for run_text in runs:
                raw = clean_one_text(str(run_text))

                # 👉 每个 run 单独送入 llama
                cleaned = run_llama(
                    raw,
                    pipe,
                    SYSTEM_PROMPT_DEFAULT,
                    args.max_new_tokens
                )

                cleaned_runs.append(cleaned)

            attr_clean[prompt_key] = {"runs": cleaned_runs}

        rec[args.field_out] = attr_clean

    save_json_any(args.out_path, fmt, records)

    print(f"✔ Saved: {args.out_path}")


if __name__ == "__main__":
    main()
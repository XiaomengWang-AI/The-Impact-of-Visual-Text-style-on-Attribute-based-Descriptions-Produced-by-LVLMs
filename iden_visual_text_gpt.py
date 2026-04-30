import os, json, time, hashlib
from pathlib import Path
from openai import OpenAI

from config import DATASET_PATH, RESULTS_PATH, PROMPTS

client = OpenAI()
MODEL_NAME = "gpt-4o-mini"

# --------- Parameters ---------
DETAIL = "low"
RECOG_MAX_OUTPUT_TOKENS = 32
COMPLETION_WINDOW = "24h"
POLL_INTERVAL_SEC = 10
# --------------------------------------


# ----------------- Utilities -----------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def upload_image_get_file_id(image_path: str) -> str:
    with open(image_path, "rb") as fp:
        fobj = client.files.create(file=fp, purpose="vision")
    return fobj.id

def load_or_build_image_file_cache(dataset, cache_path="image_file_cache.json"):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    changed = False
    for it in dataset:
        p = it["image_path"]
        if p in cache:
            continue
        if not os.path.exists(p):
            cache[p] = None
            changed = True
            continue
        cache[p] = upload_image_get_file_id(p)
        changed = True
        if changed:
            with open(cache_path, "w", encoding="utf-8") as wf:
                json.dump(cache, wf, ensure_ascii=False, indent=2)
    return cache

def make_batch_jsonl(lines, out_jsonl_path: str):
    with open(out_jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def upload_batch_file(jsonl_path: str) -> str:
    with open(jsonl_path, "rb") as fp:
        fobj = client.files.create(
            file=("input.jsonl", fp, "application/jsonl"),
            purpose="batch",
        )
    return fobj.id

def create_batch(input_file_id: str, endpoint="/v1/responses"):
    b = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=COMPLETION_WINDOW,
    )
    return b.id

def wait_batch_done(batch_id: str):
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(POLL_INTERVAL_SEC)

def download_file_text(file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)
    return str(content)

def parse_batch_output_jsonl(output_text: str):
    out = {}
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        if not cid:
            continue

        err = obj.get("error")
        resp = obj.get("response") or {}
        body = resp.get("body")

        if err or body is None:
            out[cid] = {"text": None, "error": err or "missing_body"}
            continue

        chunks = []
        for item in body.get("output", []):
            if item.get("type") != "message":
                continue
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    chunks.append(c.get("text", ""))

        out[cid] = {"text": ("\n".join(chunks)).strip(), "error": None}
    return out

def check_correctness(recog_text: str, ground_truth: str) -> bool:
    if not recog_text:
        return False
    clean_resp = recog_text.lower().strip().replace("_", " ").replace("-", " ")
    clean_gt = ground_truth.lower().strip().replace("_", " ").replace("-", " ")
    return clean_gt in clean_resp

def load_progress(results_file):
    recog_done = {}
    if not os.path.exists(results_file):
        return recog_done

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("stage") == "recog":
                    recog_done[str(obj["image_uid"])] = obj
            except:
                continue
    return recog_done

# ----------------- Recognition requests -----------------
def build_recognition_requests(dataset, img_cache, recog_done):
    reqs = []
    for it in dataset:
        image_uid = str(it["id"])
        if image_uid in recog_done:
            continue

        file_id = img_cache.get(it["image_path"])
        if not file_id:
            continue

        prompt = PROMPTS["recognition"].format(
            super_category=it["animal_type"]
        )

        reqs.append({
            "custom_id": f"recog::{image_uid}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL_NAME,
                "temperature": 0,
                "max_output_tokens": RECOG_MAX_OUTPUT_TOKENS,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "file_id": file_id, "detail": DETAIL},
                    ],
                }],
            }
        })
    return reqs


# ----------------- Main workflow -----------------
def main():
    dataset = load_dataset()

    results_file = RESULTS_PATH.format(model=MODEL_NAME)
    Path(os.path.dirname(results_file) or ".").mkdir(parents=True, exist_ok=True)

    recog_done = load_progress(results_file)

    img_cache = load_or_build_image_file_cache(dataset)

    recog_reqs = build_recognition_requests(dataset, img_cache, recog_done)

    if not recog_reqs:
        print("All recognition done.")
        return

    make_batch_jsonl(recog_reqs, "recog_batch.jsonl")
    fid = upload_batch_file("recog_batch.jsonl")
    bid = create_batch(fid)

    print("Recognition batch:", bid)

    b = wait_batch_done(bid)
    if b.status != "completed":
        print("Batch failed:", b.status)
        return

    out_text = download_file_text(b.output_file_id)
    recog_map = parse_batch_output_jsonl(out_text)

    wrote = 0
    for it in dataset:
        image_uid = str(it["id"])
        if image_uid in recog_done:
            continue

        r = recog_map.get(f"recog::{image_uid}", {})
        recog_text = r.get("text") or ""

        ok = check_correctness(recog_text, it["breed"])

        append_jsonl(results_file, {
            "stage": "recog",
            "image_uid": image_uid,
            "image_id": it["id"],
            "image_path": it["image_path"],

            # -------- Dataset metadata --------
            "animal_type": it.get("animal_type"),
            "breed": it.get("breed"),
            "font_category": it.get("font_category"),
            "font_name": it.get("font_name"),
            "size": it.get("size"),
            "position_id": it.get("position_id"),
            "text_color_name": it.get("text_color_name"),
            # ------------------------

            "is_correct": ok,
            "recog_output": recog_text,
            "recog_error": r.get("error"),
        })

        wrote += 1

    print(f"Done. Wrote {wrote} results to {results_file}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY")
    else:
        main()
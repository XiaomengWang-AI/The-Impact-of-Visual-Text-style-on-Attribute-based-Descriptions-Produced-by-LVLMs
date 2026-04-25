import os, json, time, hashlib
from pathlib import Path
from openai import OpenAI

from config import DATASET_PATH, RESULTS_PATH, PROMPTS  # PROMPTS: dict

client = OpenAI()
MODEL_NAME = "gpt-4o-mini"

# --------- Parameters (favor stable / low-drift measurement) ---------
DETAIL = "low"
RECOG_MAX_OUTPUT_TOKENS = 32
ATTR_MAX_OUTPUT_TOKENS = 64
ATTR_REPEAT = 5

COMPLETION_WINDOW = "24h"
POLL_INTERVAL_SEC = 10

ATTR_SLICE_SIZE_INIT = 500
ATTR_SLICE_SIZE_MIN = 30

TOKEN_LIMIT_SLEEP_BASE = 60
TOKEN_LIMIT_SLEEP_MAX = 15 * 60
# --------------------------------------


# ----------------- Shared helpers -----------------
def assert_unique_custom_ids(reqs):
    seen = set()
    dup = []
    for r in reqs:
        cid = r["custom_id"]
        if cid in seen:
            dup.append(cid)
        seen.add(cid)
    if dup:
        raise ValueError(f"Duplicate custom_id found: {dup[:5]}")
    
def get_image_uid(it) -> str:
    return f"{it['animal_type']}::{sha1(os.path.abspath(it['image_path']))}"

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


def load_or_build_image_file_cache(dataset, cache_path: str):
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
    """
    Batch output jsonl line schema (common):
    { custom_id, response:{status_code, body}, error }
    """
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
        status_code = resp.get("status_code")
        body = resp.get("body")

        if err or body is None:
            out[cid] = {
                "text": None,
                "status_code": status_code,
                "error": err or "missing_body",
                "raw": obj
            }
            continue

        chunks = []
        for item in body.get("output", []):
            if item.get("type") != "message":
                continue
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    chunks.append(c.get("text", ""))

        out[cid] = {
            "text": ("\n".join(chunks)).strip(),
            "status_code": status_code,
            "error": None,
            "raw": body
        }
    return out


def check_correctness(recog_text: str, ground_truth: str) -> bool:
    if not recog_text:
        return False
    clean_resp = recog_text.lower().strip().replace("_", " ").replace("-", " ")
    clean_gt = ground_truth.lower().strip().replace("_", " ").replace("-", " ")
    return clean_gt in clean_resp


# ----------------- Resume: read progress from results file -----------------
def load_progress(results_file):
    """
    Resume state:
    - recog_done: image_uid(str) -> {is_correct, ...}
    - attr_done: set((image_uid, prompt_key, rep))  rep in 0..ATTR_REPEAT-1
    """
    recog_done = {}
    attr_done = set()

    if not os.path.exists(results_file):
        return recog_done, attr_done

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            stage = obj.get("stage")
            image_uid = obj.get("image_uid")
            if image_uid is None:
                continue
            image_uid = str(image_uid)

            if stage == "recog":
                recog_done[image_uid] = obj

            elif stage == "attr":
                pk = obj.get("prompt_key")
                rep = obj.get("rep")
                if pk is not None and rep is not None:
                    try:
                        rep_i = int(rep)
                    except Exception:
                        continue
                    attr_done.add((image_uid, pk, rep_i))

    return recog_done, attr_done


# ----------------- Build batch requests -----------------
def build_recognition_requests(dataset, img_cache, recog_done):
    reqs = []
    for it in dataset:
        image_uid = get_image_uid(it)
        if image_uid in recog_done:
            continue

        image_path = it["image_path"]
        file_id = img_cache.get(image_path)
        animal_type = str(it.get("animal_type", "")).strip().lower()

        if animal_type not in {"cat", "dog"}:
            continue
        if not file_id:
            continue

        prompt = PROMPTS["recognition"].format(super_category=animal_type)
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


def build_attribute_requests(dataset, img_cache, recog_done, attr_done):
    """
    Build ATTR_REPEAT independent requests per (image, prompt_key).
    custom_id: attr::{image_uid}::{prompt_key}::r{rep}
    """
    attr_keys = [k for k in PROMPTS.keys() if k != "recognition"]
    reqs = []

    for it in dataset:
        image_uid = get_image_uid(it)
        r = recog_done.get(image_uid)
        if not r or not r.get("is_correct"):
            continue

        image_path = it["image_path"]
        file_id = img_cache.get(image_path)
        animal_type = str(it.get("animal_type", "")).strip().lower()

        if animal_type not in {"cat", "dog"}:
            continue
        if not file_id:
            continue

        for key in attr_keys:
            prompt = PROMPTS[key].format(super_category=animal_type)

            for rep in range(ATTR_REPEAT):
                if (image_uid, key, rep) in attr_done:
                    continue

                reqs.append({
                    "custom_id": f"attr::{image_uid}::{key}::r{rep}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": MODEL_NAME,
                        "temperature": 0,
                        "max_output_tokens": ATTR_MAX_OUTPUT_TOKENS,
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


# ----------------- Chunked submit (auto halve slice + stronger backoff) -----------------
def is_token_limit_exceeded(batch_dump: dict) -> bool:
    errs = (batch_dump.get("errors") or {})
    data = errs.get("data") if isinstance(errs, dict) else None
    if not data:
        return False
    for e in data:
        if (e.get("code") == "token_limit_exceeded") or ("Enqueued token limit" in (e.get("message") or "")):
            return True
    return False


def parse_attr_custom_id(cid: str):
    # attr::{image_uid}::{prompt_key}::r{rep}
    if not cid.startswith("attr::"):
        return None
    try:
        _, rest = cid.split("attr::", 1)
        left, rpart = rest.rsplit("::r", 1)
        image_uid, prompt_key = left.split("::", 1)
        rep = int(rpart)
        return image_uid, prompt_key, rep
    except Exception:
        return None


def parse_attr_output(text: str):
    """
    Expect a JSON array (attribute list). Return None on failure.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return None


def submit_attr_slices_with_auto_backoff(attr_reqs, results_file: str, work_dir: str):
    """
    - Submit in chunks
    - On token_limit_exceeded, halve slice_size and retry the same chunk
    - If slice_size is already minimal and limit still hit: sleep(backoff), then retry
    - Resume: rows already present in results_file are skipped
    """
    _, attr_done_runtime = load_progress(results_file)

    total = len(attr_reqs)
    if total == 0:
        print("[ATTR] No attribute requests to run.")
        return

    slice_size = ATTR_SLICE_SIZE_INIT
    i = 0
    part_idx = 0
    sleep_sec = TOKEN_LIMIT_SLEEP_BASE

    while i < total:
        cur = min(slice_size, total - i)
        part = attr_reqs[i:i + cur]

        jsonl_path = os.path.join(work_dir, f"attr_batch_part_{part_idx:04d}.jsonl")
        make_batch_jsonl(part, jsonl_path)

        fid = upload_batch_file(jsonl_path)
        bid = create_batch(fid, endpoint="/v1/responses")
        print(f"[ATTR] Submit part={part_idx} reqs={len(part)} slice_size={slice_size} batch={bid}")

        b = wait_batch_done(bid)
        if b.status != "completed":
            d = client.batches.retrieve(bid).model_dump()

            if b.status == "failed" and is_token_limit_exceeded(d):
                if slice_size > ATTR_SLICE_SIZE_MIN:
                    new_size = max(ATTR_SLICE_SIZE_MIN, slice_size // 2)
                    print(f"[ATTR] token_limit_exceeded. Reduce slice_size {slice_size} -> {new_size} and retry same slice.")
                    slice_size = new_size
                    continue

                sleep_now = min(sleep_sec, TOKEN_LIMIT_SLEEP_MAX)
                print(f"[ATTR] token_limit_exceeded at min slice_size. Sleep {sleep_now}s then retry same slice.")
                time.sleep(sleep_now)
                sleep_sec = min(int(sleep_sec * 1.7), TOKEN_LIMIT_SLEEP_MAX)
                continue

            print("[ATTR] Sub-batch failed (non-token-limit).")
            print("status:", b.status)
            print("errors:", json.dumps(d.get("errors"), ensure_ascii=False, indent=2))
            raise RuntimeError("attr sub-batch failed (non-token-limit)")

        sleep_sec = TOKEN_LIMIT_SLEEP_BASE
        out_text = download_file_text(b.output_file_id)
        part_map = parse_batch_output_jsonl(out_text)

        wrote = 0
        for cid, payload in part_map.items():
            parsed = parse_attr_custom_id(cid)
            if not parsed:
                continue
            image_uid, prompt_key, rep = parsed

            if (image_uid, prompt_key, rep) in attr_done_runtime:
                continue

            raw_text = payload.get("text") or ""
            parsed_out = parse_attr_output(raw_text)

            append_jsonl(results_file, {
                "stage": "attr",
                "image_uid": image_uid,
                "prompt_key": prompt_key,
                "rep": rep,
                "temperature": 0,
                "max_output_tokens": ATTR_MAX_OUTPUT_TOKENS,
                "parsed": parsed_out,
                "raw_text": raw_text,
                "error": payload.get("error"),
                "status_code": payload.get("status_code"),
            })
            attr_done_runtime.add((image_uid, prompt_key, rep))
            wrote += 1

        print(f"[ATTR] part={part_idx} completed, wrote={wrote}")

        i += cur
        part_idx += 1


# ----------------- Main pipeline (resumable) -----------------
def main():
    dataset = load_dataset()

    results_file = RESULTS_PATH.format(model=MODEL_NAME)
    Path(os.path.dirname(results_file) or ".").mkdir(parents=True, exist_ok=True)

    # Hash results path for a unique work dir tag so runs do not clobber each other's temp files
    run_tag = sha1(results_file)
    work_dir = os.path.join(os.path.dirname(results_file) or ".", f"_tmp_batches_{run_tag}")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    cache_path = os.path.join(work_dir, "image_file_cache.json")

    recog_done, attr_done = load_progress(results_file)

    img_cache = load_or_build_image_file_cache(dataset, cache_path=cache_path)

    # 1) Recognition
    recog_reqs = build_recognition_requests(dataset, img_cache, recog_done)
    assert_unique_custom_ids(recog_reqs)
    if recog_reqs:
        recog_jsonl = os.path.join(work_dir, "recog_batch.jsonl")
        make_batch_jsonl(recog_reqs, recog_jsonl)

        recog_input_file_id = upload_batch_file(recog_jsonl)
        recog_batch_id = create_batch(recog_input_file_id, endpoint="/v1/responses")
        print("Recognition batch:", recog_batch_id)

        b = wait_batch_done(recog_batch_id)
        if b.status != "completed":
            print("Recognition batch not completed:", b.status)
            d = client.batches.retrieve(recog_batch_id).model_dump()
            print("errors:", json.dumps(d.get("errors"), ensure_ascii=False, indent=2))
            return

        recog_out_text = download_file_text(b.output_file_id)
        recog_map = parse_batch_output_jsonl(recog_out_text)

        wrote = 0
        for it in dataset:
            image_uid = get_image_uid(it)
            if image_uid in recog_done:
                continue

            animal_type = str(it.get("animal_type", "")).strip().lower()

            if animal_type not in {"cat", "dog"}:
                append_jsonl(results_file, {
                    "stage": "recog",
                    "image_uid": image_uid,
                    "image_id": it.get("id"),
                    "image_path": it.get("image_path"),
                    "animal_type": animal_type,
                    "breed": it.get("breed"),
                    "is_correct": False,
                    "recog_output": None,
                    "recog_error": f"invalid_animal_type: {animal_type}",
                })
                recog_done[image_uid] = {"is_correct": False}
                wrote += 1
                continue

            cid = f"recog::{image_uid}"
            r = recog_map.get(cid)
            if r is None:
                append_jsonl(results_file, {
                    "stage": "recog",
                    "image_uid": image_uid,
                    "image_id": it["id"],
                    "image_path": it["image_path"],
                    "animal_type": animal_type,
                    "breed": it["breed"],
                    "is_correct": False,
                    "recog_output": None,
                    "recog_error": "missing_result",
                })
                recog_done[image_uid] = {"is_correct": False}
                wrote += 1
                continue

            recog_text = (r.get("text") or "").strip()
            ok = check_correctness(recog_text, it["breed"])
            append_jsonl(results_file, {
                "stage": "recog",
                "image_uid": image_uid,
                "image_id": it["id"],
                "image_path": it["image_path"],
                "animal_type": animal_type,
                "breed": it["breed"],
                "is_correct": ok,
                "recog_output": recog_text,
                "recog_error": r.get("error"),
                "status_code": r.get("status_code"),
            })
            recog_done[image_uid] = {
                "is_correct": ok,
                "recog_output": recog_text,
                "recog_error": r.get("error")
            }
            wrote += 1

        print(f"[RECOG] wrote {wrote} new results to {results_file}")
    else:
        print("[RECOG] nothing to do (all done)")

    recog_done, attr_done = load_progress(results_file)

    correct_n = sum(1 for _, r in recog_done.items() if r.get("is_correct"))
    print(f"Recognition correct (from results): {correct_n}/{len(dataset)}")

    # 2) Attributes
    attr_reqs = build_attribute_requests(dataset, img_cache, recog_done, attr_done)
    print(f"[ATTR] remaining requests: {len(attr_reqs)} (repeat={ATTR_REPEAT})")

    submit_attr_slices_with_auto_backoff(attr_reqs, results_file=results_file, work_dir=work_dir)

    print("Done. Results appended to:", results_file)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable.")
    else:
        main()
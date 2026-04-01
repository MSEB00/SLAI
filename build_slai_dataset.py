import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def clean_text(text):
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(text.split())


def add_example(examples, source, instruction, response):
    instruction = clean_text(instruction)
    response = clean_text(response)
    if not instruction or not response:
        return
    examples.append(
        {
            "source": source,
            "instruction": instruction,
            "response": response,
        }
    )


def load_dolly(limit):
    ds = load_dataset("HuggingFaceH4/databricks_dolly_15k", split="train")
    examples = []
    for row in ds:
        instruction = clean_text(row.get("instruction", ""))
        context = clean_text(row.get("input", ""))
        response = clean_text(row.get("output", ""))
        if not instruction or not response:
            continue
        if context:
            instruction = f"{instruction}\nContext: {context}"
        add_example(examples, "dolly15k", instruction, response)
        if len(examples) >= limit:
            break
    return examples


def load_helpsteer(limit, min_score):
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    examples = []
    for row in ds:
        helpfulness = float(row.get("helpfulness", 0))
        correctness = float(row.get("correctness", 0))
        coherence = float(row.get("coherence", 0))
        if min(helpfulness, correctness, coherence) < min_score:
            continue
        add_example(examples, "helpsteer2", row.get("prompt", ""), row.get("response", ""))
        if len(examples) >= limit:
            break
    return examples


def load_oasst(limit):
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    prompt_by_id = {}
    for row in ds:
        if row.get("role") != "prompter":
            continue
        if row.get("lang") != "en":
            continue
        if row.get("deleted"):
            continue
        if row.get("review_result") is False:
            continue
        prompt_text = clean_text(row.get("text", ""))
        if prompt_text:
            prompt_by_id[row.get("message_id")] = prompt_text

    examples = []
    for row in ds:
        if row.get("role") != "assistant":
            continue
        if row.get("lang") != "en":
            continue
        if row.get("deleted"):
            continue
        if row.get("review_result") is False:
            continue

        parent_id = row.get("parent_id")
        prompt_text = prompt_by_id.get(parent_id)
        response_text = clean_text(row.get("text", ""))
        if not prompt_text or not response_text:
            continue

        add_example(examples, "oasst1", prompt_text, response_text)
        if len(examples) >= limit:
            break
    return examples


def load_feedback_log(path):
    path = Path(path)
    if not path.exists():
        return []

    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            add_example(examples, "slai_feedback", row.get("user_input", ""), row.get("final_reply", ""))
    return examples


def dedupe_examples(examples):
    seen = set()
    deduped = []
    for ex in examples:
        key = (ex["instruction"].lower(), ex["response"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build SLAI SFT training dataset from public datasets + local feedback logs.")
    parser.add_argument("--output_dir", default="data", help="Output directory for JSONL files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--dolly_limit", type=int, default=1200)
    parser.add_argument("--helpsteer_limit", type=int, default=1200)
    parser.add_argument("--oasst_limit", type=int, default=1200)
    parser.add_argument("--helpsteer_min_score", type=float, default=3.0)
    parser.add_argument("--feedback_log", default="slai_feedback_log.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    all_examples = []
    all_examples.extend(load_dolly(args.dolly_limit))
    all_examples.extend(load_helpsteer(args.helpsteer_limit, args.helpsteer_min_score))
    all_examples.extend(load_oasst(args.oasst_limit))
    all_examples.extend(load_feedback_log(args.feedback_log))

    all_examples = dedupe_examples(all_examples)
    random.shuffle(all_examples)

    if not all_examples:
        raise RuntimeError("No training examples were collected.")

    valid_size = int(len(all_examples) * args.valid_ratio)
    valid_size = max(1, valid_size)
    valid_rows = all_examples[:valid_size]
    train_rows = all_examples[valid_size:]

    output_dir = Path(args.output_dir)
    train_path = output_dir / "slai_sft_train.jsonl"
    valid_path = output_dir / "slai_sft_valid.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    print(f"Built {len(all_examples)} examples total.")
    print(f"Train: {len(train_rows)} -> {train_path}")
    print(f"Valid: {len(valid_rows)} -> {valid_path}")


if __name__ == "__main__":
    main()

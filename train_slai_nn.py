import argparse
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from slai_nn import CharTokenizer, TinyGRULM


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def apply_stop_config(args):
    path = Path(args.stop_config).resolve()
    if not path.exists():
        return args

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[train] stop config ignored ({exc}): {path}")
        return args

    if not isinstance(payload, dict):
        print(f"[train] stop config ignored (not an object): {path}")
        return args

    targets = payload
    if isinstance(payload.get("training"), dict):
        targets = payload["training"]

    target_train = _safe_float(targets.get("target_train_loss"), default=None)
    target_valid = _safe_float(targets.get("target_valid_loss"), default=None)
    patience = _safe_int(targets.get("early_stop_patience"), default=None)
    min_epochs = _safe_int(targets.get("min_epochs_before_stop"), default=None)

    if target_train is not None:
        args.target_train_loss = target_train
    if target_valid is not None:
        args.target_valid_loss = target_valid
    if patience is not None and patience >= 0:
        args.early_stop_patience = patience
    if min_epochs is not None and min_epochs >= 1:
        args.min_epochs_before_stop = min_epochs

    print(
        "[train] loaded stop config:",
        f"train<={args.target_train_loss}",
        f"valid<={args.target_valid_loss}",
        f"patience={args.early_stop_patience}",
        f"min_epochs={args.min_epochs_before_stop}",
    )
    return args


def clean_text(text):
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(text.split())


def is_high_quality_pair(instruction, response, min_alpha_ratio=0.45, max_chars=900):
    if not instruction or not response:
        return False
    if len(instruction) > max_chars or len(response) > max_chars:
        return False
    if len(instruction) < 3 or len(response) < 3:
        return False
    if re.search(r"(.)\1{12,}", response):
        return False
    joined = f"{instruction} {response}"
    alpha = len(re.findall(r"[A-Za-z]", joined))
    ratio = alpha / max(1, len(joined))
    return ratio >= float(min_alpha_ratio)


def add_pair(rows, instruction, response, source):
    instruction = clean_text(instruction)
    response = clean_text(response)
    if not is_high_quality_pair(instruction, response):
        return
    rows.append(
        {
            "instruction": instruction,
            "response": response,
            "source": source,
        }
    )


def load_from_hf(args):
    from datasets import load_dataset

    rows = []
    dolly_count = 0

    dolly = load_dataset("HuggingFaceH4/databricks_dolly_15k", split="train")
    for item in dolly:
        prompt = clean_text(item.get("instruction", ""))
        context = clean_text(item.get("input", ""))
        reply = clean_text(item.get("output", ""))
        if context:
            prompt = f"{prompt}\nContext: {context}"
        add_pair(rows, prompt, reply, "dolly15k")
        dolly_count += 1
        if dolly_count >= args.max_per_dataset:
            break

    helpsteer = load_dataset("nvidia/HelpSteer2", split="train")
    hs_count = 0
    for item in helpsteer:
        helpfulness = float(item.get("helpfulness", 0))
        correctness = float(item.get("correctness", 0))
        coherence = float(item.get("coherence", 0))
        if min(helpfulness, correctness, coherence) < args.helpsteer_min_score:
            continue
        add_pair(rows, item.get("prompt", ""), item.get("response", ""), "helpsteer2")
        hs_count += 1
        if hs_count >= args.max_per_dataset:
            break

    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    prompt_by_id = {}
    for item in oasst:
        if item.get("role") != "prompter":
            continue
        if item.get("lang") != "en":
            continue
        if item.get("deleted"):
            continue
        if item.get("review_result") is False:
            continue
        prompt_text = clean_text(item.get("text", ""))
        if prompt_text:
            prompt_by_id[item.get("message_id")] = prompt_text

    oa_count = 0
    for item in oasst:
        if item.get("role") != "assistant":
            continue
        if item.get("lang") != "en":
            continue
        if item.get("deleted"):
            continue
        if item.get("review_result") is False:
            continue
        parent_id = item.get("parent_id")
        prompt_text = prompt_by_id.get(parent_id, "")
        reply_text = clean_text(item.get("text", ""))
        add_pair(rows, prompt_text, reply_text, "oasst1")
        oa_count += 1
        if oa_count >= args.max_per_dataset:
            break

    return rows


def load_local_jsonl(path, source):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = item.get("instruction") or item.get("prompt") or item.get("user_input")
            response = item.get("response") or item.get("output") or item.get("final_reply")
            add_pair(rows, instruction, response, source)
    return rows


def dedupe(rows):
    seen = set()
    result = []
    for item in rows:
        key = (
            item["instruction"].strip().lower(),
            item["response"].strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def write_pairs_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            payload = {
                "instruction": item["instruction"],
                "response": item["response"],
                "source": item.get("source", "unknown"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_dialogue_text(rows):
    texts = []
    for item in rows:
        text = f"User: {item['instruction']}\nSLAI: {item['response']}\n"
        texts.append(text)
    return texts


def build_token_stream(tokenizer, texts):
    stream = []
    for text in texts:
        stream.extend([tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id])
    return stream


class SequenceDataset(Dataset):
    def __init__(self, token_stream, seq_len):
        super().__init__()
        self.tokens = torch.tensor(token_stream, dtype=torch.long)
        self.seq_len = int(seq_len)
        self.max_start = max(1, len(self.tokens) - self.seq_len - 1)

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        start = int(idx)
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if chunk.shape[0] < self.seq_len + 1:
            pad_len = (self.seq_len + 1) - chunk.shape[0]
            chunk = torch.cat([chunk, torch.full((pad_len,), 0, dtype=torch.long)], dim=0)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    if total_count == 0:
        return float("inf")
    return total_loss / total_count


def load_compatible_state(model, checkpoint_state):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        compatible[key] = value

    if compatible:
        model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


def write_live_node_snapshot(
    model,
    sample_x,
    output_path,
    epoch,
    step,
    avg_loss,
    max_nodes=128,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        emb = model.embedding(sample_x)
        out, _ = model.rnn(emb)
        last_hidden = model.norm(out[:, -1, :]).squeeze(0).detach().float().cpu()
        proj_norm = model.proj.weight.detach().float().cpu().norm(dim=0)

    node_count = max(8, min(int(max_nodes), int(last_hidden.shape[0]), int(proj_norm.shape[0])))
    x_axis = list(range(node_count))
    hidden_vals = last_hidden[:node_count].tolist()
    proj_vals = proj_norm[:node_count].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle(f"SLAI Live Node View | epoch={epoch} step={step} avg_loss={avg_loss:.4f}")

    axes[0].plot(x_axis, hidden_vals, linewidth=1.2)
    axes[0].set_ylabel("Hidden Activation")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_axis, proj_vals, linewidth=1.2)
    axes[1].set_ylabel("Proj Weight Norm")
    axes[1].set_xlabel(f"Node Index (first {node_count})")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    internet_ok = False
    if not args.skip_internet:
        try:
            rows.extend(load_from_hf(args))
            internet_ok = True
        except Exception as exc:
            print(f"[dataset] Internet dataset load failed: {exc}")

    rows.extend(load_local_jsonl("data/slai_sft_train.jsonl", "local_sft_train"))
    rows.extend(load_local_jsonl("data/slai_sft_valid.jsonl", "local_sft_valid"))
    rows.extend(load_local_jsonl("slai_feedback_log.jsonl", "feedback"))

    rows = dedupe(rows)
    if args.max_local_rows > 0:
        rows = rows[: args.max_local_rows]
    if len(rows) < 100:
        raise RuntimeError(
            "Not enough training rows. Need at least 100 rows. "
            "Check internet connection or populate data/slai_sft_train.jsonl."
        )

    random.shuffle(rows)
    valid_size = max(64, int(len(rows) * args.valid_ratio))
    valid_size = min(valid_size, max(1, len(rows) - 1))
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]

    train_texts = format_dialogue_text(train_rows)
    valid_texts = format_dialogue_text(valid_rows)

    tokenizer = CharTokenizer().fit(train_texts + valid_texts)
    train_stream = build_token_stream(tokenizer, train_texts)
    valid_stream = build_token_stream(tokenizer, valid_texts)

    train_ds = SequenceDataset(train_stream, args.seq_len)
    valid_ds = SequenceDataset(valid_stream, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = TinyGRULM(
        vocab_size=tokenizer.vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = bool(args.mixed_precision and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    config = {
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seq_len": args.seq_len,
        "trained_with_internet_sources": internet_ok,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "profile": args.profile,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_pairs_jsonl(output_dir / "pairs.jsonl", train_rows)

    model_path = output_dir / "model.pt"
    interim_model_path = output_dir / "model_interim.pt"
    best_model_path = output_dir / "model_best.pt"
    summary_path = output_dir / "train_summary.json"
    metrics_path = output_dir / "train_metrics.jsonl"

    def write_running_summary(
        status,
        epoch_idx=0,
        step_idx=0,
        train_loss_value=None,
        valid_loss_value=None,
        best_epoch_value=None,
        best_valid_value=None,
        stop_reason="",
    ):
        payload = {
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "device": str(device),
            "rows_total": len(rows),
            "rows_train": len(train_rows),
            "rows_valid": len(valid_rows),
            "epochs_configured": int(args.epochs),
            "current_epoch": int(epoch_idx),
            "current_step": int(step_idx),
            "last_train_loss": train_loss_value,
            "last_valid_loss": valid_loss_value,
            "best_epoch": best_epoch_value,
            "best_valid_loss": best_valid_value,
            "target_train_loss": args.target_train_loss,
            "target_valid_loss": args.target_valid_loss,
            "early_stop_patience": int(args.early_stop_patience),
            "stop_reason": stop_reason,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_metric(event_type, payload):
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_running_summary(status="running")

    if args.resume and model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get("model_state", checkpoint)
            loaded_count, skipped = load_compatible_state(model, state_dict)
            model_tensor_count = len(model.state_dict())
            if loaded_count == 0:
                print(f"[train] resume skipped: no compatible tensors in {model_path}; starting fresh for current architecture.")
            else:
                print(f"[train] resumed partially from {model_path}: loaded {loaded_count} tensors, skipped {len(skipped)}.")
                # Only load optimizer state when all model tensors matched.
                # Partial model resumes can have incompatible optimizer slot shapes.
                full_model_match = len(skipped) == 0 and loaded_count == model_tensor_count
                if full_model_match and isinstance(checkpoint, dict) and checkpoint.get("optimizer_state"):
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                    except Exception:
                        pass
                else:
                    print("[train] optimizer state reset for safety (partial/incompatible resume).")
        except Exception as exc:
            print(f"[train] resume failed ({exc}); starting fresh.")

    best_valid = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    stop_reason = ""
    live_nodes_enabled = bool(args.live_nodes)
    live_nodes_path = Path(args.live_nodes_file).resolve() if args.live_nodes_file else (output_dir / "live_nodes.png")
    stop_after_epoch = False

    # Initialize best metrics from existing best checkpoint when available.
    best_source = best_model_path if best_model_path.exists() else model_path
    if best_source.exists():
        try:
            previous_best = torch.load(best_source, map_location="cpu")
            prev_valid = previous_best.get("valid_loss")
            prev_epoch = previous_best.get("epoch")
            if isinstance(prev_valid, (int, float)):
                best_valid = float(prev_valid)
                if isinstance(prev_epoch, int):
                    best_epoch = int(prev_epoch)
                print(f"[train] baseline best loaded from {best_source.name}: epoch={best_epoch}, valid_loss={best_valid:.4f}")
        except Exception:
            pass

    interrupted = False
    last_epoch = 0
    last_step = 0
    last_train_loss = None
    last_valid_loss = None
    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            model.train()
            running = 0.0
            steps = 0
            optimizer.zero_grad(set_to_none=True)
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
                    loss = loss / max(1, args.grad_accum_steps)

                scaler.scale(loss).backward()
                if (steps + 1) % max(1, args.grad_accum_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running += float(loss.item()) * max(1, args.grad_accum_steps)
                steps += 1
                last_step = steps
                if steps % max(1, args.log_every_steps) == 0:
                    avg = running / max(1, steps)
                    print(f"[train] epoch={epoch} step={steps} avg_loss={avg:.4f}", flush=True)
                    append_metric(
                        "step",
                        {
                            "epoch": epoch,
                            "step": steps,
                            "avg_loss": avg,
                        },
                    )
                    write_running_summary(
                        status="running",
                        epoch_idx=epoch,
                        step_idx=steps,
                        train_loss_value=avg,
                        valid_loss_value=None,
                        best_epoch_value=best_epoch if best_epoch > 0 else None,
                        best_valid_value=None if best_valid == float("inf") else best_valid,
                    )
                    if (
                        args.target_train_loss is not None
                        and epoch >= args.min_epochs_before_stop
                        and avg <= float(args.target_train_loss)
                    ):
                        stop_reason = f"target_train_loss_reached({avg:.4f}<={args.target_train_loss})"
                        print(f"[train] early stop triggered: {stop_reason}")
                        stop_after_epoch = True
                        break
                if steps % max(1, args.save_every_steps) == 0:
                    payload = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "valid_loss": None,
                    }
                    torch.save(payload, interim_model_path)
                    print(f"[train] interim checkpoint saved at epoch={epoch} step={steps}", flush=True)
                if live_nodes_enabled and steps % max(1, args.live_nodes_every_steps) == 0:
                    try:
                        sample_x = x[:1].detach()
                        write_live_node_snapshot(
                            model=model,
                            sample_x=sample_x,
                            output_path=live_nodes_path,
                            epoch=epoch,
                            step=steps,
                            avg_loss=(running / max(1, steps)),
                            max_nodes=args.live_nodes_max_nodes,
                        )
                        print(f"[train] live node snapshot -> {live_nodes_path}", flush=True)
                    except Exception as exc:
                        print(f"[train] live node snapshot disabled due to error: {exc}", flush=True)
                        live_nodes_enabled = False

            train_loss = running / max(1, steps)
            last_train_loss = train_loss
            valid_loss = evaluate(model, valid_loader, device)
            last_valid_loss = valid_loss
            scheduler.step()
            ppl = float(torch.exp(torch.tensor(valid_loss)).item()) if valid_loss < 20 else float("inf")
            print(
                f"[train] epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} valid_ppl={ppl:.2f}"
            )
            append_metric(
                "epoch",
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_ppl": ppl,
                },
            )

            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                no_improve_epochs = 0
                payload = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                }
                torch.save(payload, model_path)
                torch.save(payload, best_model_path)
                print(f"[train] saved best model at epoch {epoch} -> {model_path}")
            else:
                no_improve_epochs += 1

            write_running_summary(
                status="running",
                epoch_idx=epoch,
                step_idx=steps,
                train_loss_value=train_loss,
                valid_loss_value=valid_loss,
                best_epoch_value=best_epoch if best_epoch > 0 else None,
                best_valid_value=None if best_valid == float("inf") else best_valid,
            )

            if (
                args.target_valid_loss is not None
                and epoch >= args.min_epochs_before_stop
                and valid_loss <= float(args.target_valid_loss)
            ):
                stop_reason = f"target_valid_loss_reached({valid_loss:.4f}<={args.target_valid_loss})"
                print(f"[train] early stop triggered: {stop_reason}")
                break
            if stop_after_epoch:
                print("[train] stopping run after train-loss target was reached.")
                break

            if (
                args.early_stop_patience > 0
                and epoch >= args.min_epochs_before_stop
                and no_improve_epochs >= int(args.early_stop_patience)
            ):
                stop_reason = f"no_improvement_for_{no_improve_epochs}_epochs"
                print(f"[train] early stop triggered: {stop_reason}")
                break
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "keyboard_interrupt"
        print("[train] interrupted by user (KeyboardInterrupt).")
    finally:
        # Ensure model.pt points to best-known checkpoint even after interruption.
        if best_model_path.exists():
            try:
                shutil.copy2(best_model_path, model_path)
                print(f"[train] restored best checkpoint to {model_path.name} from {best_model_path.name}")
            except Exception as exc:
                print(f"[train] warning: failed to restore best checkpoint: {exc}")

    metadata = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "device": str(device),
        "rows_total": len(rows),
        "rows_train": len(train_rows),
        "rows_valid": len(valid_rows),
        "last_epoch": last_epoch,
        "last_step": last_step,
        "last_train_loss": last_train_loss,
        "last_valid_loss": last_valid_loss,
        "interrupted": interrupted,
        "stopped_early": bool(stop_reason and not interrupted),
        "stop_reason": stop_reason,
    }
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] artifacts written to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLAI local NN (GRU) from internet datasets + local logs.")
    parser.add_argument("--output-dir", default="artifacts/slai_nn")
    parser.add_argument("--max-per-dataset", type=int, default=3000)
    parser.add_argument("--helpsteer-min-score", type=float, default=3.0)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--profile", choices=["balanced", "2b_like"], default="balanced")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/model.pt when available.")
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--log-every-steps", type=int, default=120)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--live-nodes", action="store_true", help="Write live node visualization PNG during training.")
    parser.add_argument("--live-nodes-every-steps", type=int, default=400, help="Steps interval for live node snapshot.")
    parser.add_argument("--live-nodes-max-nodes", type=int, default=128, help="How many nodes to display in live snapshot.")
    parser.add_argument("--live-nodes-file", default="", help="Optional PNG path for live node snapshot.")
    parser.add_argument("--target-train-loss", type=float, default=None, help="Early-stop when running avg_loss <= this value.")
    parser.add_argument("--target-valid-loss", type=float, default=None, help="Early-stop when epoch valid_loss <= this value.")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if valid loss does not improve for N epochs (0 disables).")
    parser.add_argument("--min-epochs-before-stop", type=int, default=1, help="Minimum epochs before early-stop checks apply.")
    parser.add_argument("--stop-config", default="training_stop_config.json", help="JSON file with stop targets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-internet", action="store_true", help="Skip downloading internet datasets and use local files only.")
    parser.add_argument("--max-local-rows", type=int, default=0, help="Cap total rows after dedupe (0 means no cap).")
    args = parser.parse_args()

    if args.profile == "2b_like":
        # Practical "2b-like" defaults for 4GB VRAM GPUs.
        args.emb_dim = max(args.emb_dim, 320)
        args.hidden_dim = max(args.hidden_dim, 640)
        args.num_layers = max(args.num_layers, 3)
        args.seq_len = max(args.seq_len, 192)
        args.epochs = max(args.epochs, 6)
        args.batch_size = min(args.batch_size, 8)
        args.grad_accum_steps = max(args.grad_accum_steps, 3)
        args.lr = min(args.lr, 1.0e-3)
        args.resume = True
        args.mixed_precision = True
        args.top_k = max(args.top_k, 60)
        args.top_p = max(args.top_p, 0.93)
        args.repetition_penalty = max(args.repetition_penalty, 1.12)
        args.min_epochs_before_stop = max(args.min_epochs_before_stop, 1)

    return apply_stop_config(args)


if __name__ == "__main__":
    train(parse_args())

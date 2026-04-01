import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metrics_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def plot_architecture(config, output_png):
    emb_dim = int(config.get("emb_dim", 0))
    hidden_dim = int(config.get("hidden_dim", 0))
    num_layers = int(config.get("num_layers", 0))
    vocab_size = int(config.get("vocab_size", 0))
    seq_len = int(config.get("seq_len", 0))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("SLAI NN 2D Architecture Diagram")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, 1.4, 2.0, 1.2, f"Input Tokens\nseq_len={seq_len}"),
        (3.0, 1.4, 2.2, 1.2, f"Embedding\nvocab={vocab_size}\nemb_dim={emb_dim}"),
        (5.6, 1.4, 2.0, 1.2, f"GRU Stack\nlayers={num_layers}\nhidden={hidden_dim}"),
        (8.0, 1.4, 1.4, 1.2, "Linear Head\n(Logits)"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.5, 2.0), (3.0, 2.0)),
        ((5.2, 2.0), (5.6, 2.0)),
        ((7.6, 2.0), (8.0, 2.0)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))

    fig.tight_layout()
    fig.savefig(output_png, dpi=140)
    plt.close(fig)


def plot_training_curves(metrics_rows, output_png):
    step_x = []
    step_loss = []
    epoch_x = []
    epoch_train = []
    epoch_valid = []

    for row in metrics_rows:
        event = str(row.get("event", ""))
        if event == "step":
            step_x.append(int(row.get("step", 0)))
            step_loss.append(float(row.get("avg_loss", 0.0)))
        elif event == "epoch":
            ep = int(row.get("epoch", 0))
            epoch_x.append(ep)
            epoch_train.append(float(row.get("train_loss", 0.0)))
            epoch_valid.append(float(row.get("valid_loss", 0.0)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("SLAI NN Training Curves (2D)")
    if step_x:
        ax.plot(step_x, step_loss, label="avg_loss (step)", linewidth=1.5)
    if epoch_x:
        ax.plot(epoch_x, epoch_train, "o-", label="train_loss (epoch)", linewidth=2)
        ax.plot(epoch_x, epoch_valid, "o-", label="valid_loss (epoch)", linewidth=2)
    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create 2D matplotlib visualizations for SLAI NN.")
    parser.add_argument("--model-dir", default="artifacts/slai_nn")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    metrics_path = model_dir / "train_metrics.jsonl"
    arch_png = model_dir / "nn_architecture_2d.png"
    curve_png = model_dir / "training_curves_2d.png"

    config = load_json(config_path)
    # Best-effort vocab size from tokenizer when available.
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        try:
            tok = load_json(tok_path)
            config["vocab_size"] = len(tok.get("stoi", {}))
        except Exception:
            pass

    plot_architecture(config, arch_png)
    metrics_rows = load_metrics_jsonl(metrics_path)
    plot_training_curves(metrics_rows, curve_png)

    print(f"[viz] Wrote: {arch_png}")
    print(f"[viz] Wrote: {curve_png}")


if __name__ == "__main__":
    main()

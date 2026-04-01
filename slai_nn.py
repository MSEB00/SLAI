import json
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class CharTokenizer:
    def __init__(self, stoi=None):
        self.stoi = dict(stoi or {})
        self.itos = {index: token for token, index in self.stoi.items()}
        self.pad_id = self.stoi.get(PAD_TOKEN, 0)
        self.bos_id = self.stoi.get(BOS_TOKEN, 1)
        self.eos_id = self.stoi.get(EOS_TOKEN, 2)
        self.unk_id = self.stoi.get(UNK_TOKEN, 3)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def fit(self, texts):
        charset = sorted({char for text in texts for char in str(text)})
        tokens = list(SPECIAL_TOKENS) + [char for char in charset if char not in SPECIAL_TOKENS]
        self.stoi = {token: idx for idx, token in enumerate(tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        return self

    def encode(self, text):
        return [self.stoi.get(char, self.unk_id) for char in str(text)]

    def decode(self, ids):
        chars = []
        for idx in ids:
            token = self.itos.get(int(idx), UNK_TOKEN)
            if token in {PAD_TOKEN, BOS_TOKEN}:
                continue
            if token == EOS_TOKEN:
                break
            chars.append(token)
        return "".join(chars)

    def save(self, path):
        payload = {"stoi": self.stoi}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi=payload["stoi"])


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size, emb_dim=192, hidden_dim=384, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        x = self.embedding(input_ids)
        out, hidden = self.rnn(x, hidden)
        out = self.norm(out)
        logits = self.proj(out)
        return logits, hidden

    def generate(
        self,
        input_ids,
        max_new_tokens=120,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.08,
        eos_id=None,
    ):
        tokens = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)
            next_token_logits = logits[:, -1, :] / max(0.2, float(temperature))
            if repetition_penalty and repetition_penalty > 1.0:
                recent = tokens[:, -80:]
                for token_id in torch.unique(recent):
                    next_token_logits[:, token_id] /= float(repetition_penalty)
            if top_k and top_k > 0:
                current_top_k = min(top_k, next_token_logits.shape[-1])
                values, indices = torch.topk(next_token_logits, k=current_top_k, dim=-1)
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                next_token_logits = filtered
            if top_p and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > float(top_p)
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                sorted_logits[sorted_mask] = float("-inf")
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, sorted_indices, sorted_logits)
                next_token_logits = filtered
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_id is not None and int(next_token.item()) == int(eos_id):
                break
        return tokens


def format_messages_as_prompt(messages):
    lines = []
    for item in messages:
        role = str(item.get("role", "user")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"SLAI: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("SLAI:")
    return "\n".join(lines)


class SLAINNEngine:
    def __init__(self, model_dir="artifacts/slai_nn", max_new_tokens=160, device="auto"):
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "model.pt"
        tokenizer_path = self.model_dir / "tokenizer.json"
        config_path = self.model_dir / "config.json"
        pairs_path = self.model_dir / "pairs.jsonl"
        if not model_path.exists() or not tokenizer_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"SLAI NN artifacts missing in {self.model_dir}. Run: python train_slai_nn.py --output-dir {self.model_dir}"
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(config.get("temperature", 0.72))
        self.top_k = int(config.get("top_k", 50))
        self.top_p = float(config.get("top_p", 0.92))
        self.repetition_penalty = float(config.get("repetition_penalty", 1.1))

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = TinyGRULM(
            vocab_size=self.tokenizer.vocab_size,
            emb_dim=int(config.get("emb_dim", 192)),
            hidden_dim=int(config.get("hidden_dim", 384)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.self_learning_path = Path("self_learning_memory.jsonl")
        self._self_learning_mtime = None
        self.self_learning_pairs = []
        self.reply_pairs = []
        if pairs_path.exists():
            try:
                with pairs_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        instruction = str(item.get("instruction", "")).strip()
                        response = str(item.get("response", "")).strip()
                        if instruction and response:
                            self.reply_pairs.append((instruction, response))
                        if len(self.reply_pairs) >= 20000:
                            break
            except Exception:
                self.reply_pairs = []
        self._refresh_self_learning_pairs()

    def _extract_reply(self, generated_text):
        marker = "SLAI:"
        if marker in generated_text:
            generated_text = generated_text.split(marker)[-1]
        for stop_marker in ("\nUser:", "\nSystem:", "\nSLAI:"):
            if stop_marker in generated_text:
                generated_text = generated_text.split(stop_marker)[0]
        reply = generated_text.strip()
        if not reply:
            reply = "I am still learning. Ask me in a simpler way."
        return reply

    def _looks_unstable(self, text):
        value = str(text or "").strip()
        if len(value) < 4:
            return True
        if re.search(r"(.)\1{12,}", value):
            return True
        if len(re.findall(r"[a-zA-Z]{2,}", value)) < 2:
            return True
        return False

    def _last_user_text(self, messages):
        for item in reversed(messages):
            if str(item.get("role", "")).strip().lower() == "user":
                return str(item.get("content", "")).strip()
        return ""

    def _retrieve_reply(self, user_text):
        query = str(user_text or "").strip().lower()
        if not query:
            return None

        query_tokens = set(re.findall(r"[a-z0-9]+", query))
        if not query_tokens:
            return None

        def best_match(pairs, threshold, max_reply_len):
            best_score = 0.0
            best_reply = None
            for instruction, response in pairs:
                candidate = instruction.lower()
                if len(response) > max_reply_len:
                    continue
                ratio = SequenceMatcher(None, query, candidate).ratio()
                cand_tokens = set(re.findall(r"[a-z0-9]+", candidate))
                overlap = 0.0
                if query_tokens and cand_tokens:
                    overlap = len(query_tokens & cand_tokens) / float(len(query_tokens | cand_tokens))
                score = max(ratio, overlap)
                if score > best_score:
                    best_score = score
                    best_reply = response
            if best_score >= threshold:
                return best_reply
            return None

        # Prefer user-learned memory first.
        learned = best_match(self.self_learning_pairs, threshold=0.52, max_reply_len=420)
        if learned:
            return learned

        # Public dataset retrieval: require stricter match and shorter replies.
        return best_match(self.reply_pairs, threshold=0.72, max_reply_len=260)

    def _rule_based_reply(self, user_text):
        text = str(user_text or "").strip().lower()
        if not text:
            return None
        if "time" in text:
            now_ist = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
            return f"It's {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST."
        if "who are you" in text or "what are you" in text:
            return "I am SLAI, your self-learning AI assistant."
        if "what can you do" in text or "capabilities" in text:
            return "I can chat, remember key facts, and help with reminders and planning."
        return None

    def _refresh_self_learning_pairs(self):
        path = self.self_learning_path
        if not path.exists():
            self.self_learning_pairs = []
            self._self_learning_mtime = None
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return

        if self._self_learning_mtime is not None and mtime <= self._self_learning_mtime:
            return

        pairs = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    instruction = str(item.get("instruction", "")).strip()
                    response = str(item.get("response", "")).strip()
                    if instruction and response:
                        pairs.append((instruction, response))
            if len(pairs) > 3000:
                pairs = pairs[-3000:]
        except Exception:
            return

        self.self_learning_pairs = pairs
        self._self_learning_mtime = mtime

    def chat(self, messages, response_format=None):
        if response_format == "json":
            return {"message": {"content": "{}"}}

        self._refresh_self_learning_pairs()
        user_text = self._last_user_text(messages)
        ruled = self._rule_based_reply(user_text)
        if ruled:
            return {"message": {"content": ruled}}
        retrieved = self._retrieve_reply(user_text)
        if retrieved:
            return {"message": {"content": retrieved}}

        prompt = format_messages_as_prompt(messages)
        token_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_id=self.tokenizer.eos_id,
            )

        generated_text = self.tokenizer.decode(output[0].tolist())
        reply = self._extract_reply(generated_text)
        if self._looks_unstable(reply):
            fallback = self._retrieve_reply(user_text)
            if fallback:
                reply = fallback
            else:
                reply = "I am still learning. Please ask that in a shorter and clearer way."
        return {"message": {"content": reply}}

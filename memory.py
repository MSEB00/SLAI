import json
import re
from pathlib import Path

MEMORY_FILE = Path(__file__).with_name("memory.json")
VALID_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
GENERIC_KEYS = {"key", "value", "fact", "memory", "field", "data", "user_fact"}
LEGACY_VALUE_PATTERN = re.compile(r"^\s*([^,=]+)\s*,\s*value\s*=\s*(.+)\s*$", re.IGNORECASE)


def normalize_key(raw_key):
    if raw_key is None:
        return None

    key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key).strip("_")

    if not key or key in GENERIC_KEYS:
        return None
    if not VALID_KEY_PATTERN.fullmatch(key):
        return None
    return key


def normalize_value(raw_value):
    if raw_value is None:
        return None

    value = re.sub(r"\s+", " ", str(raw_value).strip())
    if not value or len(value) > 200:
        return None
    return value


def normalize_entry(raw_key, raw_value):
    key = normalize_key(raw_key)
    value = normalize_value(raw_value)
    if key and value:
        return key, value

    if value:
        legacy = LEGACY_VALUE_PATTERN.match(value)
        if legacy:
            legacy_key = normalize_key(legacy.group(1))
            legacy_value = normalize_value(legacy.group(2))
            if legacy_key and legacy_value:
                return legacy_key, legacy_value

    return None


def load_memory():
    if not MEMORY_FILE.exists():
        return {}

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    normalized = {}
    for raw_key, raw_value in data.items():
        normalized_entry = normalize_entry(raw_key, raw_value)
        if not normalized_entry:
            continue
        key, value = normalized_entry
        normalized[key] = value

    if normalized != data:
        save_memory(normalized)
    return normalized


def save_memory(memory_data):
    temp_file = MEMORY_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=4, ensure_ascii=False)
    temp_file.replace(MEMORY_FILE)


def add_fact(fact_key, fact_value):
    normalized_entry = normalize_entry(fact_key, fact_value)
    if not normalized_entry:
        return
    key, value = normalized_entry

    memory_data = load_memory()
    memory_data[key] = value
    save_memory(memory_data)


def get_memory_context():
    memory_data = load_memory()
    if not memory_data:
        return "No known user facts yet."

    lines = ["Known facts about the user:"]
    for key, value in memory_data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

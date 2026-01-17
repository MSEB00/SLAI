# memory.py
import json
import os

MEMORY_FILE = 'memory.json'

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def remember_entity(entity_type, value):
    memory = load_memory()
    if entity_type not in memory:
        memory[entity_type] = []
    if value not in memory[entity_type]:
        memory[entity_type].append(value)
    save_memory(memory)

def recall_entity(entity_type):
    memory = load_memory()
    return memory.get(entity_type, [])

def recall_all():
    return load_memory()

# neural_net.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


MODEL_DIR = "transformer_intent_model"
DATASET_PATH = "intent_dataset.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        base_data = {
            "intents": [
                {"text": "open notepad", "intent": "open_app"},
                {"text": "open whatsapp", "intent": "open_app"},
                {"text": "what time is it", "intent": "time"},
                {"text": "tell me a joke", "intent": "joke"},
                {"text": "hello", "intent": "greet"},
                {"text": "exit", "intent": "exit"}
            ]
        }
        with open(DATASET_PATH, 'w') as f:
            json.dump(base_data, f, indent=2)
    with open(DATASET_PATH, 'r') as f:
        return json.load(f)

def save_dataset(dataset):
    with open(DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)

def add_training_example(text, intent):
    dataset = load_dataset()
    dataset["intents"].append({"text": text, "intent": intent})
    save_dataset(dataset)

def train_and_save():
    dataset = load_dataset()
    texts = [item["text"] for item in dataset["intents"]]
    intents = [item["intent"] for item in dataset["intents"]]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(label_encoder.classes_))
    model.to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 4
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model and label encoder
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save label encoder classes
    with open(f"{MODEL_DIR}/label_encoder.json", 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)

def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    with open(f"{MODEL_DIR}/label_encoder.json", 'r') as f:
        classes = json.load(f)
    return tokenizer, model, classes

def predict_intent(text):
    if not os.path.exists(MODEL_DIR):
        print("Training model for first time. This may take a while...")
        train_and_save()
    tokenizer, model, classes = load_model()
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    intent = classes[pred]
    return intent

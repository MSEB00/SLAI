import face_recognition
import os
import numpy as np
import pickle

# Folder with subfolders (person names) containing images
dataset_path = "faces"
embeddings = []
labels = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) == 0:
            print(f"[!] No face found in {image_path}")
            continue

        embeddings.append(face_encodings[0])
        labels.append(person_name)

# Save to disk for training
with open("face_data.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)

print("[✓] Face embeddings saved to face_data.pkl")

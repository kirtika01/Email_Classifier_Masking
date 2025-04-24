import os
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = "trained_model/cnn_classifier.pt"
VECTORIZER_PATH = "trained_model/vectorizer.pkl"
MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


class EmailDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        features = self.vectorizer.transform([text]).toarray()
        return {
            "features": torch.FloatTensor(features).squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.fc_input_dim = 32 * (input_dim // 4)

        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class EmailClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_epoch(self, model, data_loader, optimizer):
        model.train()
        total_loss = 0
        predictions = []
        actual = []

        for batch in tqdm(data_loader, desc="Training"):
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            actual.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(predictions) == np.array(actual))
        return total_loss / len(data_loader), accuracy

    def train_model(self, X: List[str], y: List[str]) -> Dict[str, float]:
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=MAX_SEQUENCE_LENGTH)
            self.vectorizer.fit(X)

        unique_labels = sorted(set(y))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.rev_label_map = {label: i for i, label in self.label_map.items()}

        y_encoded = [self.rev_label_map[label] for label in y]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        train_dataset = EmailDataset(X_train, y_train, self.vectorizer)
        val_dataset = EmailDataset(X_val, y_val, self.vectorizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        self.model = CNNClassifier(MAX_SEQUENCE_LENGTH, len(unique_labels))
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        best_val_accuracy = 0
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_loss, train_acc = self.train_epoch(
                self.model, train_loader, optimizer
            )

            self.model.eval()
            val_predictions = []
            val_actual = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    features = batch["features"].to(self.device)
                    labels = batch["label"]

                    outputs = self.model(features)
                    val_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_actual.extend(labels.numpy())

            val_accuracy = np.mean(np.array(val_predictions) == np.array(val_actual))

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()

        return {
            "train_accuracy": train_acc,
            "test_accuracy": val_accuracy,
            "best_test_accuracy": best_val_accuracy,
        }

    def predict_category(self, text: str) -> str:
        self.load_model_and_vectorizer()
        self.model.eval()
        features = self.vectorizer.transform([text]).toarray()
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            predicted_class = outputs.argmax(dim=1).item()
            return self.label_map[predicted_class]

    def save_model(self) -> None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)

        import joblib

        joblib.dump(self.vectorizer, VECTORIZER_PATH)

        label_map_path = os.path.join(os.path.dirname(MODEL_PATH), "label_map.npy")
        np.save(label_map_path, self.label_map)

    def load_model_and_vectorizer(self) -> None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file not found")

        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Vectorizer file not found")

        label_map_path = os.path.join(os.path.dirname(MODEL_PATH), "label_map.npy")
        self.label_map = np.load(label_map_path, allow_pickle=True).item()
        self.rev_label_map = {v: k for k, v in self.label_map.items()}

        import joblib

        self.vectorizer = joblib.load(VECTORIZER_PATH)

        self.model = CNNClassifier(MAX_SEQUENCE_LENGTH, len(self.label_map))
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.to(self.device)
        self.model.eval()

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm

from models import EmailClassifier
from utils import load_data, mask_pii, preprocess_text

def prepare_data() -> Tuple[List[str], List[str]]:
    print("\n1. Loading dataset...")
    df = load_data()
    print(f"Dataset loaded: {len(df)} emails")
    
    print("\n2. Processing emails...")
    processed_emails = []
    for i, email in enumerate(tqdm(df['email'], desc="Email processing")):
        masked_email, entities = mask_pii(email)
        processed_email = preprocess_text(masked_email)
        processed_emails.append(processed_email)
        
        if i < 2:
            print(f"\nSample {i+1} processed email (truncated):")
            print(f"Original: {email[:100]}...")
            print(f"Processed: {processed_email[:100]}...")
    
    return processed_emails, df['type'].tolist()

def evaluate_model(classifier: EmailClassifier, texts: List[str], labels: List[str], num_examples: int = 5) -> None:
    print("\nGenerating predictions...")
    predictions = []
    for text in tqdm(texts, desc="Predicting"):
        prediction = classifier.predict_category(text)
        predictions.append(prediction)
    
    print("\nClassification Report:")
    report = classification_report(labels, predictions)
    print(report)
    
    print("\nConfusion Matrix:")
    unique_labels = sorted(list(set(labels)))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    
    print("\n" + " " * 10 + " ".join(f"{label:<10}" for label in unique_labels))
    for i, row in enumerate(cm):
        print(f"{unique_labels[i]:<10}" + " ".join(f"{cell:<10}" for cell in row))
    
    print("\nExample Predictions (by class):")
    shown_classes = set()
    for i in range(len(texts)):
        if labels[i] not in shown_classes and len(shown_classes) < len(unique_labels):
            print(f"\nClass: {labels[i]}")
            print(f"Text: {texts[i][:100]}...")
            print(f"Predicted: {predictions[i]}")
            shown_classes.add(labels[i])

def main():
    os.makedirs("trained_model", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\n1. Loading and preparing data...")
    X, y = prepare_data()
    
    print("\n2. Initializing CNN model...")
    print("\nModel Configuration:")
    print("- Model: CNN with TF-IDF features")
    print("- Max Features:", 1000)
    print("- Batch Size:", 32)
    print("- Learning Rate:", 0.001)
    print("- Epochs:", 10)
    print("- Dropout Rate:", 0.3)
    print("- Conv Layers:", "2 (64 & 32 filters)")
    print("- Device:", "cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = EmailClassifier()
    metrics = classifier.train_model(X, y)
    
    print("\n3. Final Results:")
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Best Test Accuracy: {metrics['best_test_accuracy']:.4f}")
    
    if metrics['train_accuracy'] - metrics['test_accuracy'] > 0.1:
        print("\n⚠️ Warning: Model shows signs of overfitting")
    
    print("\n4. Performing final evaluation...")
    eval_classifier = EmailClassifier()
    eval_classifier.load_model_and_vectorizer()
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test set size: {len(X_test)} emails")
    
    evaluate_model(eval_classifier, X_test, y_test)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
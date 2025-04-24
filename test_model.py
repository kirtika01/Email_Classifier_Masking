from models import EmailClassifier
from utils import preprocess_text, mask_pii

def test_single_email():
    classifier = EmailClassifier()
    classifier.load_model_and_vectorizer()
    
    print("\nEnter the email text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        elif lines:
            break
    
    email_text = "\n".join(lines)
    
    print("\nPreprocessing steps:")
    print("1. PII Masking:")
    masked_email, entities = mask_pii(email_text)
    print(f"After PII masking:\n{masked_email}\n")
    
    print("2. NLP Preprocessing:")
    processed_email = preprocess_text(masked_email)
    print(f"After NLP preprocessing:\n{processed_email}\n")
    
    print("\nPredicting category...")
    prediction = classifier.predict_category(processed_email)
    
    print("\nClassification Results:")
    print(f"Predicted Category: {prediction}")
    
    print("\nDetected PII Entities:")
    for entity in entities:
        print(f"- {entity['classification']}: {entity['entity']}")
        
    print("\nNLP Analysis:")
    print("- Removed stop words")
    print("- Performed lemmatization")
    print("- Kept only nouns, verbs, adjectives, and adverbs")
    print("- Normalized named entities")

if __name__ == "__main__":
    print("Email Classification Test Tool")
    print("============================")
    test_single_email()
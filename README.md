# Email Classification System with PII Masking

A FastAPI-based system that masks Personally Identifiable Information (PII) in emails and classifies them into predefined support categories.

## Features

- PII Detection and Masking using:
  - Named Entity Recognition (spaCy)
  - Regular Expressions
  - Custom text processing
- Email Classification using Machine Learning
- REST API with FastAPI
- Ready for Hugging Face Spaces deployment

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd email_classifier_project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
email_classifier_project/
│
├── app.py               # FastAPI application entry point
├── api.py              # API endpoints definition
├── models.py           # ML model functionality
├── utils.py            # PII masking and preprocessing
├── requirements.txt    # Project dependencies
├── README.md           # This file
├── data/              
│   └── emails.csv      # Training data
└── trained_model/      # Saved model files
```

## Usage

1. Start the server:
```bash
python app.py
```

2. The API will be available at `http://localhost:8000`

3. API Documentation will be available at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### POST /classify

Classifies an email and masks PII.

Request body:
```json
{
    "email_body": "Your email text here"
}
```

Response:
```json
{
    "input_email_body": "original email text",
    "list_of_masked_entities": [
        {
            "position": [start_index, end_index],
            "classification": "entity_type",
            "entity": "original_entity_value"
        }
    ],
    "masked_email": "masked email text",
    "category_of_the_email": "predicted category"
}
```


## Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Upload the project files
3. Set the Space SDK to "Gradio/FastAPI"
4. Configure the environment variables if needed
5. Deploy the application

## Training Custom Models

To train the model with your own data:

1. Place your labeled email dataset in `data/emails.csv`
2. Format should be: text column containing email body, label column containing categories
3. Modify the training parameters in `models.py` if needed
4. Run training script (will be provided separately)


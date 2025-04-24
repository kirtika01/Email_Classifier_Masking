# Email Classifier API with PII Masking

This repository contains a FastAPI-based email classification service that masks PII (Personally Identifiable Information) and classifies emails into categories. The service is containerized and ready for deployment on Hugging Face Spaces.

## API Endpoints

- `POST /classify`: Classifies an email and masks PII

## Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Set the visibility as needed

2. Upload the repository files to the Space:
   - Upload all files including the Dockerfile
   - Make sure the trained model files are included in the `trained_model/` directory

3. The Space will automatically build and deploy the Docker container

## API Usage

```python
import requests

# Replace with your Spaces URL
API_URL = "https://your-username-space-name.hf.space"

# Example request
response = requests.post(
    f"{API_URL}/classify",
    json={"email_body": "Your email content here"}
)
print(response.json())
```

## Environment

- Python 3.9
- FastAPI
- Docker
- Runs on port 7860 (default for Hugging Face Spaces)

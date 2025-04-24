from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json

from utils import mask_pii, preprocess_text
from models import EmailClassifier

app = FastAPI(
    title="Email Classifier API",
    description="API for classifying emails with PII masking",
    version="1.0.0",
)

classifier = EmailClassifier()


class EmailRequest(BaseModel):
    email_body: str


class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str


class ClassificationResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str


@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest) -> Dict[str, Any]:
    try:
        email_body = request.email_body
        masked_email, entities = mask_pii(email_body)
        processed_email = preprocess_text(masked_email)
        category = classifier.predict_category(processed_email)
        print(f"Predicted category: {category}")

        response = {
            "input_email_body": email_body,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category,
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


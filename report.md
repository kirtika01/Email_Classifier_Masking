# Email Classification System with PII Masking
## Technical Implementation Report

### 1. Introduction

This report details the implementation of an email classification system with PII (Personally Identifiable Information) masking capabilities. The system is designed to process incoming emails, detect and mask sensitive information, and classify the emails into predefined support categories using machine learning techniques.

### 2. PII Detection and Masking

#### 2.1 Named Entity Recognition (NER)
- Implemented using spaCy's en_core_web_sm model
- Detects entities such as:
  - PERSON (names)
  - ORG (organizations)
  - GPE (locations)
  - CARDINAL (numbers)

#### 2.2 Regular Expression Patterns
Custom regex patterns were implemented to capture:
- Email addresses: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}`
- Phone numbers: Various formats including international
- Social Security Numbers: Standard 9-digit format with optional hyphens

#### 2.3 Masking Process
1. Combine NER and regex detections
2. Sort entities by position to handle overlapping
3. Replace entities with type-specific masks (e.g., [PERSON], [EMAIL])
4. Track position changes for accurate metadata

### 3. Machine Learning Implementation

#### 3.1 Model Selection
- Chosen model: RandomForestClassifier
- Reasons for selection:
  - Good performance on text classification
  - Handles high-dimensional sparse data well
  - Less prone to overfitting
  - Interpretable results

#### 3.2 Text Preprocessing
1. Convert to lowercase
2. Remove special characters
3. Normalize whitespace
4. TF-IDF vectorization with 5000 features

#### 3.3 Training Process
- Train/test split: 80/20
- Feature engineering using TfidfVectorizer
- Model parameters:
  - n_estimators: 100
  - random_state: 42 for reproducibility

### 4. API Implementation

#### 4.1 FastAPI Framework
- RESTful API design
- Async endpoint handling
- Automatic API documentation
- Input validation using Pydantic models

#### 4.2 Endpoints
1. POST /classify
   - Handles email classification requests
   - Returns masked text and predictions
2. GET /health
   - System health monitoring
   - Basic status checks

### 5. Challenges and Solutions

1. **Entity Overlap**
   - Challenge: Multiple PII detections overlapping
   - Solution: Sorting and sequential replacement

2. **Performance Optimization**
   - Challenge: Slow processing of long emails
   - Solution: Efficient regex compilation and spaCy pipeline optimization

3. **Model Serialization**
   - Challenge: Consistent model loading across deployments
   - Solution: Implemented robust joblib serialization

4. **Error Handling**
   - Challenge: Various input formats and edge cases
   - Solution: Comprehensive input validation and error handling

### 6. Conclusion

The implemented system successfully combines NLP techniques and machine learning to create a robust email processing pipeline. The modular design allows for easy updates and maintenance, while the FastAPI implementation ensures efficient handling of requests at scale.

Future improvements could include:
- Implementation of more sophisticated ML models
- Additional PII detection patterns
- Performance optimizations for large-scale deployment
- Integration with email processing systems

The system is ready for deployment on Hugging Face Spaces and can be easily integrated into existing email processing workflows.
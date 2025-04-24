# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install build dependencies, Python packages and clean up
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY api.py .
COPY app.py .
COPY models.py .
COPY utils.py .
COPY trained_model/ ./trained_model/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Start the FastAPI application with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
FROM python:3.11-slim

# Metadata for HuggingFace Spaces
LABEL space_sdk="docker"
LABEL tags="openenv"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make sure Python path is set
ENV PYTHONPATH=/app

# Default command: run baseline
CMD ["python", "agent/baseline.py"]

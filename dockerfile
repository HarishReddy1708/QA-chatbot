# Use official Python image (Debian-based)
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies for PyTorch, NLTK, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . .

# Expose port (adjust if needed)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

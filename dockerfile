# Use official Python image (Debian-based)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded NLTK data
COPY nltk_data /usr/share/nltk_data

# Set NLTK environment variable
ENV NLTK_DATA=/usr/share/nltk_data

# Copy application files
COPY . .

# Expose Flask app port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]

FROM python:3.12.9-slim

WORKDIR /app

# Install system dependencies for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model file
COPY ./model/genre_classification_fma.keras .

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "src/app.py", "0.0.0.0:5000", "app:app"]

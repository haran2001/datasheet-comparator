# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set environment variables
# Prevents Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]

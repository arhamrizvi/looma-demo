FROM python:3.11-slim

# System deps (optional but safe)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY insights_api.py .

# Expose port
EXPOSE 8080

# Cloud Run expects the server on port 8080
ENV PORT=8080

CMD ["uvicorn", "insights_api:app", "--host", "0.0.0.0", "--port", "8080"]
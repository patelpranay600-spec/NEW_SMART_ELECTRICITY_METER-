FROM python:3.11-slim

WORKDIR /app

# Install system build essentials
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependency specification and install Python libraries
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate dataset splits and train the neural network artifact during container build
RUN python -m scripts.prepare_splits || true
RUN python -m scripts.train || true

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Run both FastAPI (uvicorn) and Streamlit concurrently
CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]
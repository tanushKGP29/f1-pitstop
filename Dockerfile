# Use a standard, clean Python slim image
FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some GBDT wheel setups)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Expose ports for both services
# 8000: FastAPI, 8501: Streamlit
EXPOSE 8000
EXPOSE 8501

# Default startup command (starts Streamlit dashboard)
# To run FastAPI instead, override the CMD with:
# uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

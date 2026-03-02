FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgdal-dev gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.app.txt .
RUN pip install -r requirements.app.txt
COPY shared/inference_jobs.py ./inference_jobs.py
COPY shared/__init__.py ./shared/__init__.py
ENV PYTHONPATH=/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
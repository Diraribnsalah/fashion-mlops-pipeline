FROM python:3.10-slim

 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

 
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# open the port 8000
EXPOSE 8000

# run the FastApi application 
CMD ["uvicorn", "src.model_inference:app", "--host", "0.0.0.0", "--port", "8000"]

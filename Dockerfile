FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/api.py src/
COPY models/phase4_logistic_regression.pkl models/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
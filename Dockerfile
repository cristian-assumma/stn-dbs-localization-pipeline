# 1. Base Image: Python Slim 
FROM python:3.10-slim

# 2. Optimization to prevent useless cache usage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Work directory
WORKDIR /app

# 4. Copying requirements and clean installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copying src code
COPY . .

# 6. Entrypoint dynamic CLI
ENTRYPOINT ["python", "main.py"]
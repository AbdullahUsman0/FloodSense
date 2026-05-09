FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.docker.txt ./

RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "main.py", "--server.port=8502", "--server.address=0.0.0.0"]
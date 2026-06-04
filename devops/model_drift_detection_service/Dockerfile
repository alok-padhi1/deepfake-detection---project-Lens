FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requests psycopg2-binary kafka-python

COPY drift_detector.py .

CMD ["python", "drift_detector.py"]

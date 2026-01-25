FROM python:3.12-slim

RUN pip install pika

WORKDIR /app
COPY autoincrement.py .
COPY logging_utils.py .
ENTRYPOINT ["python", "autoincrement.py"]


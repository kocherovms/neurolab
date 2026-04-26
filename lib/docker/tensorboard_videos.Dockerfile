FROM python:3.12-slim

WORKDIR /app

RUN pip install flask

COPY tensorboard_videos.py .

EXPOSE 5000

CMD ["python", "tensorboard_videos.py"]
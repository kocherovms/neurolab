FROM python:3.12-slim

WORKDIR /app

RUN pip install flask

COPY lang_utils.py tensorboard_videos.py .

EXPOSE 5000

CMD ["python", "tensorboard_videos.py"]
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    python3 \
    python3-pip \
    tzdata \
    curl \
    docker.io \
    docker-buildx \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Moscow

RUN pip install --break-system-packages --no-cache-dir pika  
RUN pip install --break-system-packages --no-cache-dir boto3 
RUN pip install --break-system-packages --no-cache-dir certifi # req-d for boto3 to work (otherwise boto3 will complain about self signed cert)

WORKDIR /app
COPY launch_dispatcher.py lang_utils.py logging_utils.py  .
#CMD ["python3", "launch_dispatcher.py"]
ENTRYPOINT ["python3", "launch_dispatcher.py"]

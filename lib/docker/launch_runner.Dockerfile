FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y python3-pip tzdata netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Moscow

RUN pip install --break-system-packages --no-cache-dir boto3 
RUN pip install --break-system-packages --no-cache-dir certifi # req-d for boto3 to work (otherwise boto3 will complain about self signed cert)
RUN pip install --break-system-packages --no-cache-dir docker 
RUN pip install --break-system-packages --no-cache-dir names_generator 

WORKDIR /app
COPY launch_runner.py lang_utils.py logging_utils.py command_listener.py .

# Command scripts
USER 0

RUN echo "#!/bin/sh" >> /usr/bin/drain
RUN echo "echo drain | nc localhost 5555" >> /usr/bin/drain
RUN chmod +x /usr/bin/drain

# CMD ["python3", "launch_runner.py"]
ENTRYPOINT ["python3", "launch_runner.py"]

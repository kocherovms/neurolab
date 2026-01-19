FROM torch-cpu:2.9.1

# https://stackoverflow.com/questions/77331227/fontconfig-error-no-writable-cache-directories
RUN apt-get update
RUN apt-get install -y libnss-unknown
RUN chmod -R 777 /home
ENV HOME=/home

RUN pip install pika matplotlib
WORKDIR /app
COPY metrics_collector.py .
ENTRYPOINT ["python", "metrics_collector.py"]


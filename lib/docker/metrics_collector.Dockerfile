FROM torch-cpu:2.9.1
RUN pip install pika matplotlib
WORKDIR /app
COPY metrics_collector.py .
ENTRYPOINT ["python", "metrics_collector.py"]


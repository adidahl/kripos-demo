FROM python:3.9-slim

RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary \
    boto3

WORKDIR /mlflow

COPY mlflow-server.sh /mlflow/
RUN chmod +x /mlflow/mlflow-server.sh

EXPOSE 5000

CMD ["/mlflow/mlflow-server.sh"] 
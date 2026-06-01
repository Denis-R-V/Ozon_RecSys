FROM ghcr.io/mlflow/mlflow:v3.12.0

# Pre-install S3 + Postgres drivers so the container starts instantly
# (no pip install at runtime → faster startup, reproducible image)
RUN pip install --quiet --no-cache-dir boto3 psycopg2-binary

EXPOSE 5000

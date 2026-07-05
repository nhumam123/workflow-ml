FROM python:3.12-slim

# Install dependencies yang dibutuhkan
RUN pip install --no-cache-dir mlflow==2.19.0

WORKDIR /app

# Salin folder mlruns Anda agar model terbaca di dalam container
COPY mlruns /app/mlruns

EXPOSE 5005

# Jalankan perintah mlflow serve di dalam container (host wajib 0.0.0.0)
CMD ["mlflow", "models", "serve", "-m", "runs:/ff94c8fbea874263a15009458b1a0fc3/model", "--env-manager=local", "--host", "0.0.0.0", "-p", "5005"]
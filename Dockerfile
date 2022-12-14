FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt --no-cache-dir

EXPOSE 5000

ENTRYPOINT ["python"]
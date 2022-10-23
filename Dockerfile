FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt --no-cache-dir

EXPOSE 5000

CMD gunicorn -b 0.0.0.0:5000 --timeout 120 --access-logfile - "user.app:create_app()

#ENTRYPOINT ["gunicorn"]
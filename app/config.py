import os
from dotenv import load_dotenv
from pathlib import Path
from decouple import config

#env_path = Path('.') / '.env'
#load_dotenv(dotenv_path=env_path)


#AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
#AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
#DATABASE_URI = os.getenv("DATABASE_URI")
#BUCKET_S3 = os.getenv("BUCKET_S3")
#URL_S3 = os.getenv("URL_S3")
#REGION_S3 = os.getenv("REGION_S3")
#REGION_S3_SOUTH = os.getenv("REGION_S3_SOUTH")
#MYSQL_USER = os.getenv("MYSQL_USER")
#MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
#MYSQL_HOST = os.getenv("MYSQL_HOST")

AWS_ACCESS_KEY = config("AWS_ACCESS_KEY")
AWS_SECRET_KEY = config("AWS_SECRET_KEY")
#DATABASE_URI = config("DATABASE_URI")
BUCKET_S3 = config("BUCKET_S3")
URL_S3 = config("URL_S3")
REGION_S3 = config("REGION_S3")
REGION_S3_SOUTH = config("REGION_S3_SOUTH")
MYSQL_USER = config("MYSQL_USER")
MYSQL_PASSWORD = config("MYSQL_PASSWORD")
MYSQL_HOST = config("MYSQL_HOST")
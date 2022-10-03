from json import JSONEncoder
import os
import timm
import numpy as np
from flask import Flask
from werkzeug.utils import secure_filename
from PIL import Image
from fastai import *
from flask import request
from fastai.vision.all import *
from fastai.imports import *
from fastai.vision import *
import pymysql
from flask import jsonify
import boto3
import io
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy import update
import base64
import uuid
import config

print(config.DATABASE_URI)
BUCKET_S3 = config.BUCKET_S3

s3 = boto3.client(
    "s3",
    aws_access_key_id=config.AWS_ACCESS_KEY,
    aws_secret_access_key=config.AWS_SECRET_KEY
)

app = Flask(__name__)

MAX_RETRIES_NUM = 10
no_host_available = True
retry_tries = 0
'''
while no_host_available:
    try:
        app.config['SQLALCHEMY_DATABASE_URI']= config.DATABASE_URI
        break
    except:
        retry_tries += 1
        if retry_tries == MAX_RETRIES_NUM:
                sys.exit()
        time.sleep(5)
'''
app.config['SQLALCHEMY_DATABASE_URI']= f"mysql+pymysql://{config.MYSQL_USER}:{config.MYSQL_PASSWORD}@{config.MYSQL_HOST}:3306/flaskmysql"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False

db = SQLAlchemy(app)
ma = Marshmallow(app)


class FireImage(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    key = db.Column(db.String(255), nullable=False)
    image = db.Column(db.String(255), nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(255), nullable=False)
    taken_at = db.Column(db.DateTime, nullable=False)
    camera_type = db.Column(db.String(255), nullable=False)
    id_camera = db.Column(db.String(255), nullable=False)
    false_alarm = db.Column(db.Boolean, nullable=False)
    zone = db.Column(db.Integer, nullable=False)

    def __init__(self, image, longitude, latitude, prediction, taken_at, id_camera, camera_type, false_alarm, zone, key):
        self.image = image
        self.longitude = longitude
        self.latitude = latitude
        self.prediction = prediction
        self.taken_at = taken_at
        self.id_camera = id_camera
        self.camera_type = camera_type
        self.false_alarm = false_alarm
        self.zone = zone
        self.key = key


db.create_all()    

class FireImageSchema(ma.Schema):
    class Meta:
        fields = ('id', 'image', 'longitude', 'latitude', 'taken_at', "id_camera", "camera_type", "false_alarm", "zone", "key")

fire_image_schema = FireImageSchema()
fire_images_schema = FireImageSchema(many=True)

def setup_learner():
    #await download_file(export_file_url, path / export_file_name)
    path = "./app/models/model_convnext_small_in22k_version_1.pkl"
    try:
        learn = load_learner(path)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


learn = setup_learner()

def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a fake file stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file as a argument, passing a bytes io ins
  image.save(imgByteArr, format="JPEG")
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def upload_file_to_s3(file, bucket_name):
    """
    Docs: http://boto3.readthedocs.io/en/latest/guide/s3.html
    """
    try:
        s3.upload_fileobj(
            file,
            bucket_name,
            file.filename
        )
    except Exception as e:
        print("Something Happened: ", e)

@app.route("/", methods=["GET"])
def index():
    return jsonify({'result': "Fire detection app"})


@app.route('/analyze', methods=['POST'])
def analyze():
    latitude = request.form["latitude"]
    longitude = request.form["longitude"]
    img_data = request.files['file'].read()
    img_data_ = request.files['file'].filename
    print(img_data_)
    taken_at = request.form["taken_at"]
    id_camera = request.form["id_camera"]
    camera_type = request.form["camera_type"]
    zone = request.form["zone"]
    false_alarm = False
    taken_at = datetime.strptime(taken_at, '%d/%m/%y %H:%M:%S')
    image = PILImage.create(io.BytesIO(img_data))
    extension = img_data_.split(".")[-1]
    key = str(uuid.uuid1())+"." + extension
    prediction = learn.predict(image)[0]
    if prediction == "fire":
        s3.upload_fileobj(
            io.BytesIO(img_data),
            BUCKET_S3,
            key,
        )
        url = "%s%s" % (config.URL_S3, str(key))
        new_alert = FireImage(
            image=url, 
            latitude=latitude, 
            longitude=longitude, 
            prediction=prediction, 
            taken_at=taken_at, 
            camera_type=camera_type,
            false_alarm=false_alarm,
            id_camera=id_camera,
            zone=zone,
            key=key
            )
        db.session.add(new_alert)
        db.session.commit()

    return jsonify({'result': str(prediction)})

def new_read_image_from_s3():
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(BUCKET_S3)
    for object_summary in my_bucket.objects:
        print(object_summary)


def read_image_from_s3(bucket, key, region_name=config.REGION_S3):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3
    """
    s3 = boto3.resource('s3',
                aws_access_key_id=config.AWS_ACCESS_KEY,
                aws_secret_access_key=config.AWS_SECRET_KEY, 
                region_name=config.REGION_S3_SOUTH)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    print(file_stream)
    im = Image.open(file_stream)
    #return np.array(im)
    return list(file_stream)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@app.route('/alerts', methods=['GET'])
def get_all_fire_alerts():
    #result = s3.list_objects(Bucket=BUCKET_S3)
    all_alerts = FireImage.query.filter_by(false_alarm=0).all()
    all_alerts = fire_images_schema.dump(all_alerts)
    retrieve_images = []
    """
    for alert in all_alerts:
        for o in result.get('Contents'):
            data = o.get("Key")
            data = s3.get_object(Bucket=BUCKET_S3, Key=o.get('Key'))
            content = data['Body'].read()
            #content = content.decode("utf-8")
            data_t = type(content)
            print(data_t)
            print(content)
            #decoded = content.decode('utf-8')
            alert["image"] = content
    return jsonify(all_alerts)
    """
    #for alerts in all_alerts:
    #    image = read_image_from_s3(BUCKET_S3, alerts["image"])
    #    new_read_image_from_s3()
    #    #alerts["image"] = json.dumps(image, cls=NumpyArrayEncoder)
    #    print(type(alerts["image"]))
    #    retrieve_images.append(alerts)

    return jsonify({"data":all_alerts})
    #return jsonify({'result': contents})

@app.route('/alert/', methods=['PUT'])
def false_positive_change():
    req = request.args.keys()
    key = request.args.get('key')
    update = FireImage.query.filter_by(key=key).first()
    update.false_alarm = 1
    db.session.commit()
    #alert = update(FireImage)
    #alert = alert.values({"false_alarm": 0})
    #alert = alert.where(FireImage.id == id)
    #db.session.execute(alert)
    return jsonify({"result": "updated to false positive"})


if __name__ == '__main__':
    #if 'serve' in sys.argv:
    app.run(port=5000, debug=False)

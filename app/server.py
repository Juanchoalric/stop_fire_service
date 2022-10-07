import os
import timm
import numpy as np
from flask import Flask
from fastai import *
import pathlib
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
import uuid
from decouple import config

BUCKET_S3 = config("BUCKET_S3")
AWS_ACCESS_KEY = config("AWS_ACCESS_KEY")
AWS_SECRET_KEY = config("AWS_SECRET_KEY")
URL_S3 = config("URL_S3")
REGION_S3 = config("REGION_S3")
REGION_S3_SOUTH = config("REGION_S3_SOUTH")
MYSQL_USER = config("MYSQL_USER")
MYSQL_PASSWORD = config("MYSQL_PASSWORD")
MYSQL_HOST = config("MYSQL_HOST")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']= f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/flaskmysql"
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
    zone = db.Column(db.String(255), nullable=False)

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
    path = f"{pathlib.Path(__file__).parent.resolve()}" + "/models/model_convnext_small_in22k_version_1.pkl"
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

@app.route("/", methods=["GET"])
def index():
    return jsonify({'result': "Fire detection app"})


@app.route('/analyze', methods=['POST'])
def analyze():
    latitude = request.form["latitude"]
    longitude = request.form["longitude"]
    img_data = request.files['file'].read()
    img_data_ = request.files['file'].filename
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
        try:
            s3.upload_fileobj(
                io.BytesIO(img_data),
                BUCKET_S3,
                key,
            )
        except Exception as e:
            print("Something Happened when uploading the image: ", e)
        url = "%s%s" % (URL_S3, str(key))

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


@app.route('/alerts', methods=['GET'])
def get_all_fire_alerts():
    all_alerts = FireImage.query.filter_by(false_alarm=0).all()
    all_alerts = fire_images_schema.dump(all_alerts)

    return jsonify({"data":all_alerts})


@app.route('/alert/', methods=['PUT'])
def false_positive_change():
    key = request.args.get('key')
    update = FireImage.query.filter_by(key=key).first()
    update.false_alarm = 1
    db.session.commit()

    return jsonify({"result": "updated to false positive"})


if __name__ == '__main__':
    app.run(port=5000, debug=False, host="0.0.0.0")

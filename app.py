from flask import Flask, jsonify, request, render_template, flash, redirect, url_for, Response, session
from torchvision.transforms import Compose, Resize, ToTensor
from werkzeug.utils import secure_filename
from model import CNNForNaturalDataset
from PIL import Image
import numpy as np
import requests
import io
import torch
import cv2
import os
import shutil
import math

 #Setup cấu hình cho camera
start_camera = 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Tên các class dự đoán
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
#Gọi model
model = CNNForNaturalDataset()
checkpoint = torch.load("./trained_model/best.pt")
model.load_state_dict(checkpoint['model'])
model.eval()
#Tạo Flask app
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOW_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.secret_key = "trannhancoder"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.'in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSIONS
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/', methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if os.path.isdir("static/uploads"):
            shutil.rmtree("static/uploads")
        os.mkdir("static/uploads")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash("Image Successfully Uploaded, Display And Predict Below")
        resp = requests.post("http://localhost:5000/predict",
                             files={"file": open('./static/uploads/{}'.format(filename), 'rb')})
        result = resp.json()
        return render_template('index.html', filename=filename, resp=resp.json())
    else:
        flash("Allowed image types are - png, jpg, jpeg")
        return redirect(request.url)
@app.route('/display/<filename>')
def display_images(filename):
    return redirect(url_for("static", filename="uploads/" + filename))
def transform_image(image_bytes):
    transform = Compose([
        Resize((128, 128)),
        ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0) # Thêm chiều mới vòa tensor, 0: thêm chiều đầu tiên [1, C, H, W]
def get_prediction(image_bytes):
    img_tensor = transform_image(image_bytes)
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        output = model(img_tensor)
        result = softmax(output)
    idx = torch.argmax(result)
    acc = torch.max(result)
    acc = math.ceil(acc * 100)
    return idx.item(), classes[idx], acc
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name, acc = get_prediction(img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name, 'acc': acc})

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    img = cv2.resize(img, (128, 128))/255
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img
@app.route('/camera')
def camera():
    return render_template("camera.html")
def generate_frames_camera():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed_camera')
def video_feed_camera():
    return Response(generate_frames_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/camera', methods=["POST"])
def start_or_stop():
    global start_camera
    global cap
    if start_camera == 0:
        start_camera = 1
        cap.release()
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        start_camera = 0
    return render_template('camera.html')


@app.route('/video')
def video():
    return render_template('video.html')
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    file = request.files['file']
    filename = secure_filename(file.filename)
    session['filename'] = filename
    if os.path.isdir("static/uploads"):
        shutil.rmtree("static/uploads")
    os.mkdir("static/uploads")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('predict_video.html', filename=filename)
def generate_frames_video(filename):
    if os.path.isdir("static/result"):
        shutil.rmtree("static/result")
    os.mkdir("static/result")
    video = cv2.VideoCapture("static/uploads/" + filename)
    softmax = torch.nn.Softmax()
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 480))
            img = frame.copy()
            img_save = frame.copy()
            img = preprocess_img(img)
            with torch.no_grad():
                outputs = model(img)
                result = softmax(outputs)
            idx = torch.argmax(result)
            acc = torch.max(result)
            acc = math.ceil(acc * 100)
            frame = cv2.putText(frame, "ID: {}".format(idx), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame = cv2.putText(frame, "Class Name: {} ({}%)".format(classes[idx], acc), (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed_video')
def video_feed_video():
    filename = session.get('filename')
    return Response(generate_frames_video(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
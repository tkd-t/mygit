import numpy as np
import os
import io
import time
import cv2
import base64
import dlib
import pprint

from flask import *
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
         # とりあえずサイズは小さくする
        raw_img = cv2.resize(img, (IMAGE_WIDTH, int(IMAGE_WIDTH*img.shape[0]/img.shape[1])))

        # サイズだけ変えたものも保存する
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        cv2.imwrite(raw_img_url, raw_img)
        
        frame = mask(img)
        # 加工したものを保存する
        mask_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'after_'+filename)
        cv2.imwrite(mask_img_url, frame)

        # 保存したファイルに対してエンコード
        with open(mask_img_url, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        # レスポンスのjsonに箱詰め
        response = []
        # response.append({'id':json['id'], 'result' : img_base64})
        response.append({'result' : img_base64})

        return make_json_response("ok", response)

        # return render_template('index.html', raw_img_url=raw_img_url, mask_img_url=mask_img_url)

    else:
        return redirect(url_for('index'))

def mask(img):
    res_labels = ['NO MASK!!', 'ok']
    save_dir = './live'

    model = load_model('mask_model.h5')
    detector = dlib.get_frontal_face_detector()

    red = (0,0,255)
    green = (0,255,0)
    fid = 1

    image = img
    dets = detector(image, 1)

    for k,d in enumerate(dets):
        x1 = int(d.left())
        y1 = int(d.top())
        x2 = int(d.right())
        y2 = int(d.bottom())
        print(x1,x2,x2,y2)
        
        im = image[y1:y2, x1:x2]
        im = cv2.resize(im, (50,50))
        im = im.reshape(-1, 50, 50, 3)
        
        res = model.predict([im])[0]
        v= res.argmax()
        
        color = green if v==1 else red
        border = 2 if v==1 else 5
        cv2.rectangle(image,
                    (x1, y1),(x2,y2),color,
                    thickness = border)

        cv2.putText(image,
                res_labels[v], (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, thickness=1)

    frame = cv2.resize(image, (IMAGE_WIDTH, int(IMAGE_WIDTH*img.shape[0]/img.shape[1])))
    return frame

def make_json_response(status, response):
    res = {
        'status': status,
        'response': response
    }
    return make_response(jsonify(res))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
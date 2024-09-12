from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

categories = ['Cat','Dog']
model1 = load_model("Model.h5")
folder = "D:/Gundu/DeepLearning_Projects/Dog-Cat_Classification/train"
# folder = "D:\Gundu\DeepLearning_Projects\Dog-Cat_Classification\train"
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/home',methods=['POST'])
def home():
    file = request.form['file']
    # if(file==""):
        # file = "Empty"
    img = Image.open(file)
    # img.save(folder+"/"+file)
    # img = image.load_img(img_path, target_size=(32, 32))
    img = img.resize((32,32))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds=model1.predict(x)
    pred = np.argmax(preds, axis=-1)
    print(pred)
    print(categories[pred[0]])
    # return <h1>categories[pred[0]]</h1>
    return render_template("output.html",output = categories[pred[0]])
    # return render_template("output.html",output=file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
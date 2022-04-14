from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import argparse
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
model = tf.keras.models.load_model('Model')

conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(100, 100, 3))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    classes = ['Car', 'Fish', 'Pigeon']
        image = request.files.get('image').read()
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        image = tf.image.resize(image, (100, 100))
        image = tf.expand_dims(image, axis=0)
        image = conv_base.predict(image)
                
        preds = model.predict(image)
        preds = tf.Variable(preds)
        probas = tf.keras.activations.softmax(preds).numpy().reshape(-1)
        prediction = np.argmax(probas)
        probas = np.round_(probas * 100, decimals=2)
        result = {
            'prediction': classes[prediction],
            'probas': {str(class_): str(proba) for class_ , proba in zip(classes, probas)}
        }

        return jsonify(result), 200



@app.route('/statistic', methods=['GET'])
def statistic():   
    result = {'confusion_matrix': {}} 
    classes = ['Car', 'Fish', 'Pigeon']
    precisions = [.97, 1, .98]    
    recalls = [1, .98, .98]
    conf_mat = np.array([[223, 0, 1],[2, 219, 3], [5, 0, 219]])

    for class_, precision, recall, vector in zip(classes, precisions, recalls, conf_mat):
        result[class_] = {'precision': str(precision), 'recall': str(recall)}
        result['confusion_matrix'][class_] = str(vector)

    return jsonify(result), 200
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port number') 

    args = parser.parse_args()
    app.run(port=args.port)

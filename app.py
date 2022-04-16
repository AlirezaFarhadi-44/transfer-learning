from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import argparse


app = Flask(__name__)
model = tf.keras.models.load_model('Model2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    classes = ['Car', 'Fish', 'Pigeon']
    try:
        image = request.files.get('image').read()
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.image.resize(image, (160, 160))
        image = tf.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        preds = model.predict(image)
        preds = tf.Variable(preds)
        prob = tf.keras.activations.softmax(preds).numpy().reshape(-1)
        prediction = np.argmax(prob)
        prob = np.round_(prob * 100, decimals=2)
        result = {
            'prediction': classes[prediction],
            'confidance': {str(class_): str(proba) for class_ , proba in zip(classes, prob)}
        }

        return jsonify(result), 200
        
    except Exception as ex:
        result = {'errer'}
        return jsonify(result), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port number') 

    args = parser.parse_args()
    app.run(port=args.port)

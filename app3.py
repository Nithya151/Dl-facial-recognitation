#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the saved model
model_path = "C:/Users/DELL/Downloads/FacialExpressionDataset/models/facial_expression.model.h5"  
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.resize(gray_image, (48, 48))  # Resize to match model's expected input
    gray_image = gray_image / 255.0  # Normalize pixel values
    gray_image = np.expand_dims(gray_image, -1)  # Add channel dimension
    gray_image = np.expand_dims(gray_image, 0)  # Add batch dimension
    return gray_image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)
        prediction = model.predict(img)
        emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'] 
        result = emotions[np.argmax(prediction)]
        return jsonify({'emotion': result})
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/')
def index():
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)


# In[5]:


get_ipython().run_line_magic('tb', '')


# In[ ]:





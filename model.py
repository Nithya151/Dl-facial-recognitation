#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset_dir = 'C:/Users/DELL/Downloads/FacialExpressionDataset/dataset' 
print("files in the dataset:")
for file in os.listdir(dataset_dir):
    print(file)


# In[3]:


def custom_preprocessing(img):
    # resize image
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    
    # gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Histogram Equalization
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale image
        img = cv2.equalizeHist(img)
    else:  # Color image
        for i in range(3):  # Apply histogram equalization to each channel
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    
    return img

def preprocessing_function(image):
    # Convert tensor to numpy array
    image_array = np.array(image, dtype=np.uint8)
    # Apply custom preprocessing
    processed_image = custom_preprocessing(image_array)
    # Ensure the output is float32 to match Keras' expectations
    return processed_image.astype(np.float32)

# initializing ImageDataGenerator with the custom preprocessing function
train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocessing_function,
    horizontal_flip=True, # Image Augumentation
    validation_split=0.2  
)


# Set paths and parameters
batch_size = 32
target_size = (64, 64)

train_dataset = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_dataset = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# In[4]:


# Load Efficient net 
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
#define the input with input shape
input_tensor = Input(shape=(64, 64, 3))

#model building
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)


# In[5]:


# compile the model
model.compile(optimizer = Adam(learning_rate=0.0001),
             loss='categorical_crossentropy',
              metrics = ['accuracy'])


# In[6]:


model.summary()


# In[7]:


# Train the model on your dataset
import time
start_time =time.time()

history = model.fit(
        train_dataset,
        epochs=10,
        validation_data= validation_dataset)

training_time = time.time() - start_time
print("Training time: {training_time:.2f} secs")


# In[8]:


# unfreeze the last 10 layers
for layer in base_model.layers[:-10]:
    layer.trainable = False
# add custom layers
x= base_model.output
x= GlobalAveragePooling2D()(x)
x= Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

#creaye the final model
model = Model(inputs = base_model.input, outputs = predictions)


# In[9]:


# compile the model
model.compile(optimizer=Adam(learning_rate = 0.0001),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])


# In[10]:


model.summary()


# In[11]:


# train the model 
start_time = time.time()
history = model.fit(
           train_dataset,
           epochs=10,
          validation_data = validation_dataset)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")


# In[17]:


model.save('C:/Users/DELL/Downloads/FacialExpressionDataset/facial_expression.model.h5')


# In[18]:


print(f"Training time: {training_time:.2f} secs")


# In[19]:


predictions = model.predict(validation_dataset, steps=np.ceil(validation_dataset.samples/validation_dataset.batch_size))
predicted_classes = np.argmax(predictions, axis=1)
true_labels = validation_dataset.classes

    


# In[20]:


# Calculate ROC AUC
import tensorflow as tf
roc_auc = roc_auc_score(tf.keras.utils.to_categorical(true_labels, num_classes=6), predictions, multi_class='ovr')
print(f"ROC AUC: {roc_auc:.3f}")


# In[21]:


class_names = ['Ahegao'
,'Angry'
,'Happy'
,'Neutral'
,'Sad'
,'Surprise']


# In[26]:


from sklearn.metrics import confusion_matrix as cm
predictions = model.predict(validation_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate confusion matrix
conf_matrix = cm(true_labels, predicted_labels)  # Note: 'conf_matrix' is used here
print("Confusion Matrix:", conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  # 'conf_matrix' is correctly referenced here
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# Task                                  Sub-task                                Comments
# Data Preprocessing                   Scaling and Resizing                       Done
#                                      Image Augmentation                         Done
#                                      Train and test data handled correctly      Done
#              Gaussian Blur, Histogram Equalization and Intensity thresholds     Done
# Model Trained                        Training Time?                             Done
#                                     AUC and Confusion Matrix Computed           Done
#                             Overfitting/Underfitting checked and handled       NOT Done
# Empirical Tuning                    Interpretability Implemented                None 
# 

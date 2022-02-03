#!/usr/bin/env python
# coding: utf-8

# In[62]:


import shutil
shutil.rmtree("c:/users/janneke/anaconda3/lib/site-packages/~atplotlib",  ignore_errors=True)
shutil.rmtree("c:/users/janneke/anaconda3/lib/site-packages/~atplotlib-3.2.0.dist-info",  ignore_errors=True)
shutil.rmtree("c:/users/janneke/anaconda3/lib/site-packages/~ensorflow",  ignore_errors=True)


# In[1]:


get_ipython().system('pip install pillow==8.1.0')
get_ipython().system('pip install -U matplotlib')
get_ipython().system('pip install numpy==1.19.3')
get_ipython().system('pip install opencv-python==4.5.1.48')
get_ipython().system('pip install tqdm==4.56.0')
get_ipython().system('pip install requests==2.25.1')

get_ipython().system('pip install mediapipe==0.8.3')


# In[4]:


get_ipython().system('pip install tensorflow')


# In[5]:


import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[28]:


import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests

class PoseClassificationVisualizer(object):
  """Keeps track of claassifcations for every frame and renders them."""

  def __init__(self,
               class_name,
               plot_location_x=0.05,
               plot_location_y=0.05,
               plot_max_width=0.4,
               plot_max_height=0.4,
               plot_figsize=(9, 4),
               plot_x_max=None,
               plot_y_max=None,
               counter_location_x=0.85,
               counter_location_y=0.05,
               counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
               counter_font_color='red',
               counter_font_size=0.15):
    self._class_name = class_name
    self._plot_location_x = plot_location_x
    self._plot_location_y = plot_location_y
    self._plot_max_width = plot_max_width
    self._plot_max_height = plot_max_height
    self._plot_figsize = plot_figsize
    self._plot_x_max = plot_x_max
    self._plot_y_max = plot_y_max
    self._counter_location_x = counter_location_x
    self._counter_location_y = counter_location_y
    self._counter_font_path = counter_font_path
    self._counter_font_color = counter_font_color
    self._counter_font_size = counter_font_size

    self._counter_font = None

    self._pose_classification_history = []
    self._pose_classification_filtered_history = []

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count):
    """Renders pose classifcation and counter until given frame."""
    # Extend classification history.
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)

    # Output frame with classification plot and counter.
    output_img = Image.fromarray(frame)

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    # Draw the plot.
    img = self._plot_classification_history(output_width, output_height)
    img.thumbnail((int(output_width * self._plot_max_width),
                   int(output_height * self._plot_max_height)),
                  Image.ANTIALIAS)
    output_img.paste(img,
                     (int(output_width * self._plot_location_x),
                      int(output_height * self._plot_location_y)))

    # Draw the count.
    output_img_draw = ImageDraw.Draw(output_img)
    if self._counter_font is None:
      font_size = int(output_height * self._counter_font_size)
      font_request = requests.get(self._counter_font_path, allow_redirects=True)
      self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)
    output_img_draw.text((output_width * self._counter_location_x,
                          output_height * self._counter_location_y),
                         str(repetitions_count),
                         font=self._counter_font,
                         fill=self._counter_font_color)

    return output_img

  def _plot_classification_history(self, output_width, output_height):
    fig = plt.figure(figsize=self._plot_figsize)

    for classification_history in [self._pose_classification_history,
                                   self._pose_classification_filtered_history]:
      y = []
      for classification in classification_history:
        if classification is None:
          y.append(None)
        elif self._class_name in classification:
          y.append(classification[self._class_name])
        else:
          y.append(0)
      plt.plot(y, linewidth=7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Classification history for `{}`'.format(self._class_name))
    plt.legend(loc='upper right')

    if self._plot_y_max is not None:
      plt.ylim(top=self._plot_y_max)
    if self._plot_x_max is not None:
      plt.xlim(right=self._plot_x_max)

    # Convert plot to image.
    buf = io.BytesIO()
    dpi = min(
        output_width * self._plot_max_width / float(self._plot_figsize[0]),
        output_height * self._plot_max_height / float(self._plot_figsize[1]))
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img


# In[29]:


csvs_out_train_path = 'yoga_poses_csvs_out.csv'
csvs_out_test_path = 'yoga_test_poses_csvs_out.csv'


# In[30]:


def load_pose_landmarks(csv_path):
  """Loads a CSV created by MoveNetPreprocessor.
  
  Returns:
    X: Detected landmark coordinates (33 * 3)
    y: Ground truth labels of shape (N, label_count)
    classes: The list of all class names found in the dataset
    dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
      truth labels (y) to use later to train a pose classification model.
  """

  # Load the CSV file
  dataframe = pd.read_csv(csv_path, header=None)
  df_to_process = dataframe.copy()

  # Drop the file_name columns as you don't need it during training.
  df_to_process = df_to_process.iloc[: , 1:]

  # # Extract the list of class names
  # classes = df_to_process.pop('class_name').unique()

  # Extract the labels
  y = df_to_process.pop(df_to_process.columns[0])

  # Convert the input features and labels into the correct format for training.
  X = df_to_process.astype('float32')
  y = keras.utils.to_categorical(y)

  return X, y, dataframe


# In[31]:


# Load the train data
X, y, _ = load_pose_landmarks(csvs_out_train_path)

# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.15, shuffle = True)
print(X_train)
print(X_train.shape)
print(X_val.shape)
print(y_train)
print(y_val.shape)


# In[32]:


from sklearn.utils import shuffle

# Load the test data
X_test, y_test, df_test = load_pose_landmarks(csvs_out_test_path)
# X_test, y_test = shuffle(X_test, y_test)

print(X_test)
print(y_test)


# In[33]:


# # Define the model
inputs = tf.keras.Input(shape=(99))
# # embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(2048, activation=tf.nn.relu6)(inputs)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(1024, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(512, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(256, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(32, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(16, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
outputs = keras.layers.Dense(5, activation="softmax")(layer)


model = keras.Model(inputs, outputs)
model.summary()

print(model.layers[0].weights)
print(model.layers[1].weights)
print(model.layers[2].weights)
print(model.layers[3].weights)
print(model.layers[4].weights)
print(model.layers[5].weights)


# # layer = keras.layers.Dense(128, activation=tf.nn.relu6)(inputs)
# layer = keras.layers.Dense(128)(inputs)
# layer = keras.layers.Dropout(0.2)(layer)
# layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
# layer = keras.layers.Dropout(0.2)(layer)
# outputs = keras.layers.Dense(5, activation="softmax")(layer)


# In[34]:


import numpy as np
class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)
    landmarks = np.float32(landmarks)

    return landmarks

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)
    # print(landmarks.shape)

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    print("center: ")
    print(pose_center)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier).astype("float32")
    print("pose size: ")
    print(pose_size)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it easier to debug.
    # landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """

    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]
    

    # Hips center.
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips).astype("float32")

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)


  def _get_average_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return self._get_distance(lmk_from, lmk_to)

  def _get_distance(self, lmk_from, lmk_to):
    return lmk_to - lmk_from


# In[35]:


import pandas as pd

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                              patience=20)


pose_embedder = FullBodyPoseEmbedder()
booly = 0
landmarks = X_train.to_numpy().astype("float32")
validation = X_val.to_numpy().astype("float32")
val_landmarks = []
norm_landmarks = []
for x in landmarks:
  y = np.reshape(x,[-1,3])
  embedding = pose_embedder(y)
  norm_landmarks.append(embedding)
for z in validation:
  p = np.reshape(z,[-1,3])
  embedding = pose_embedder(p)
  val_landmarks.append(embedding)

norm_landmarks = np.array(norm_landmarks)
norm_landmarks = np.reshape(norm_landmarks, [-1,99])
val_landmarks = np.array(val_landmarks)
val_landmarks = np.reshape(val_landmarks, [-1,99])
print(norm_landmarks)
print(y_train)
# Start training
history = model.fit(norm_landmarks, y_train,
                    epochs=200,
                    batch_size=16,
                    shuffle = True,
                    validation_data=(val_landmarks, y_val),
                    callbacks=[checkpoint, earlystopping],
                    verbose = 2)

print(model.layers[1].weights)


# In[36]:


# Visualize the training history to see whether you're overfitting.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['TRAIN', 'VAL'], loc='lower right')
plt.show()


# In[37]:


test_landmarks = X_test.to_numpy()
norm_test_landmarks = []
for x in test_landmarks:
  y = np.reshape(x,[-1,3])
#   print(y)
  embedding = pose_embedder(y)
  print(embedding)
  norm_test_landmarks.append(embedding)

norm_test_landmarks = np.array(norm_test_landmarks)
norm_test_landmarks = np.reshape(norm_test_landmarks, [-1,99])
# print(norm_test_landmarks)
# Evaluate the model using the TEST dataset
loss, accuracy = model.evaluate(norm_test_landmarks, y_test)


# In[38]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """Plots the confusion matrix."""
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(norm_test_landmarks)
rounded_pred = np.argmax(y_pred, axis=-1)
for i in y_pred:
  print(i)

class_names = ["goddess", "warrior2", "tree", "plank", "downdog"]

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plot_confusion_matrix(cm,
                      class_names,
                      title ='Confusion Matrix of Pose Classification Model')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_true_label,
                                                          y_pred_label))


# In[14]:


# !mkdir -p saved_model
model.save('saved_model/my_model')

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()


with open('finalModel2048.tflite', 'wb') as f:
  f.write(tflite_model)

print(model.layers[0].weights)
print(model.layers[1].weights)
print(model.layers[2].weights)
print(model.layers[3].weights)
print(model.layers[4].weights)
print(model.layers[5].weights)


# In[ ]:





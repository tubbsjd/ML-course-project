#adapted from https://github.com/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.py
#https://keras.io/examples/vision/3D_image_classification/#train-model

#data from https://openneuro.org/datasets/ds000030/versions/1.0.0

#to get the phenotype data
#wget https://openneuro.org/crn/datasets/ds000030/snapshots/1.0.0/files/participants.tsv

#R Code for Downloading the nii data
#
#R
#
#dat <- read.table("participants.tsv", header = T)
#str(dat)
#
#for(i in dat$participant_id) {
#    system(paste0("wget https://openneuro.org/crn/datasets/ds000030/snapshots/1.0.0/files/", i, ":anat:", i, "_T1w.nii.gz"))
#}
#
##to perform skullstripping and standardization
#fls <- list.files(pattern = "w.nii.gz")
#head(fls)
#
##Brain Extraction
#for(i in fls) {
#    system(paste0("/home/tubbsjd/fsl/fsl/bin/bet ", i, " ", sub(".nii.gz", "_bet.nii.gz", i)))
#}
#
##Reference Brain Alignment
#fls <- list.files(pattern = "bet.nii.gz")
#for(i in fls) {
#    system(paste0("/home/tubbsjd/fsl/fsl/bin/flirt -in ", i, " -ref /home/tubbsjd/fsl/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz -out ", sub("_bet.nii.gz", "_flirt.nii.gz", i)))
#}
#
#q(save = "no")

###python code

#python3

import os
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

"""
## Loading data and preprocessing

The files are provided in Nifti format with the extension .nii.gz. To read the scans, we use the `nibabel` package.
MRI voxel intensities depend on machine type and scanning parameters, so we need to normalize the images.
see https://blog.tensorflow.org/2018/07/an-introduction-to-biomedical-image-analysis-tensorflow-dltk.html

To process the data, we do the following:

* We first rotate the volumes by 90 degrees, so the orientation is fixed
* We scale the voxel intensity values to have a mean 0 and variance 1.
* We resize width, height and depth.

Here we define several helper functions to process the data. These functions
will be used when building training and validation datasets.
"""

import nibabel as nib
from scipy import ndimage

#from https://github.com/DLTK/DLTK/blob/dev/dltk/io/preprocessing.py#L9
def normalize(image):
  """Whitening. Normalises image to zero mean and unit variance."""
  image = image.astype(np.float32)
  mean = np.mean(image)
  std = np.std(image)
  if std > 0:
    ret = (image - mean) / std
  else:
    ret = image * 0.
  return ret


def read_nifti_file(filepath):
  """Read and load volume"""
  # Read file
  scan = nib.load(filepath)
  # Get raw data
  scan = scan.get_fdata()
  return scan


def resize_volume(img):
  """Resize across z-axis"""
  # Set the desired depth
  desired_depth = 64
  desired_width = 128
  desired_height = 128
  # Get current depth
  current_depth = img.shape[-1]
  current_width = img.shape[0]
  current_height = img.shape[1]
  # Compute depth factor
  depth = current_depth / desired_depth
  width = current_width / desired_width
  height = current_height / desired_height
  depth_factor = 1 / depth
  width_factor = 1 / width
  height_factor = 1 / height
  # Rotate
  img = ndimage.rotate(img, 90, reshape=False)
  # Resize across z-axis
  img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
  return img


def process_scan(path):
  """Read and resize volume"""
  # Read scan
  volume = read_nifti_file(path)
  # Normalize
  volume = normalize(volume)
  # Resize width, height and depth
  volume = resize_volume(volume)
  return volume


"""
Read the paths of the MRIs
"""
data = pd.read_csv("participants.tsv", sep='\t')
data.head()

#there is some problem where subj 10299, 10428, 10501, 10971, do not have a T1 image, remove them
data = data[data.participant_id != 'sub-10299']
data = data[data.participant_id != 'sub-10428']
data = data[data.participant_id != 'sub-10501']
data = data[data.participant_id != 'sub-10971']
data = data[data.participant_id != 'sub-11121']
data = data[data.participant_id != 'sub-70035']
data = data[data.participant_id != 'sub-70036']

hc = data[data['diagnosis']=='CONTROL']
scz = data[data['diagnosis']=='SCHZ']
adhd = data[data['diagnosis']=='ADHD']


#try with balanced case-control
#hc = hc[50:100]

wd=os.getcwd()

# HC scans
normal_scan_paths = [
  wd+'/'+x+':anat:'+x+'_T1w_flirt.nii.gz'
  for x in hc.participant_id.tolist()
]

# SCZ scans
scz_scan_paths = [
  wd+'/'+x+':anat:'+x+'_T1w_flirt.nii.gz'
  for x in scz.participant_id.tolist()
]

# ADHD scans
adhd_scan_paths = [
  wd+'/'+x+':anat:'+x+'_T1w_flirt.nii.gz'
  for x in adhd.participant_id.tolist()
]

print("Healthy Controls: " + str(len(normal_scan_paths)))
print("SCZ Patients: " + str(len(scz_scan_paths)))
print("ADHD Patients: " + str(len(adhd_scan_paths)))


"""
## Build train and validation datasets
Read the scans from the class directories and assign labels. Downsample the scans to have
shape of 128x128x64. Rescale the raw intensities.
Lastly, split the dataset into train and validation subsets.
"""

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
scz_scans = np.array([process_scan(path) for path in scz_scan_paths])
adhd_scans = np.array([process_scan(path) for path in adhd_scan_paths])


# Assign categorical dummy variables for class membership
# encode class values as integers
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

tmp = np.array(["HC", "SCZ", "ADHD"])
tmp2 = np.repeat(tmp, [len(normal_scans), len(scz_scans), len(adhd_scans)], axis=0)

encoder = LabelEncoder()
encoder.fit(tmp2)
encoded_Y = encoder.transform(tmp2)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Split data in the ratio 70-30 for training and validation.
nhc = len(normal_scan_paths)
nscz = len(scz_scan_paths)
nadhd = len(adhd_scan_paths)

#get class-based y
normal_y = dummy_y[:nhc]
scz_y = dummy_y[(nhc):(nhc+nscz)]
adhd_y = dummy_y[(nhc+nscz):(nhc+nscz+nadhd)]

x_train = np.concatenate((normal_scans[:round(nhc*0.7)], scz_scans[:round(nscz*0.7)], adhd_scans[:round(nadhd*0.7)]), axis=0)
y_train = np.concatenate((normal_y[:round(nhc*0.7)], scz_y[:round(nscz*0.7)], adhd_y[:round(nadhd*0.7)]), axis=0)

x_val = np.concatenate((normal_scans[round(nhc*0.7):], scz_scans[round(nscz*0.7):], adhd_scans[round(nadhd*0.7):]), axis=0)
y_val = np.concatenate((normal_y[round(nhc*0.7):], scz_y[round(nscz*0.7):], adhd_y[round(nadhd*0.7):]), axis=0)

print(
  "Number of samples in train and validation are %d and %d."
  % (x_train.shape[0], x_val.shape[0])
)

"""
## Data augmentation

The CT scans also augmented by rotating at random angles during training. Since
the data is stored in rank-3 tensors of shape `(samples, height, width, depth)`,
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on
the data. The new shape is thus `(samples, height, width, depth, 1)`. There are
different kinds of preprocessing and augmentation techniques out there,
this example shows a few simple ones to get started.
"""

import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""
    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


"""
While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.
"""

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

"""
Visualize an augmented MRI scan.
"""

import matplotlib.pyplot as plt

data = train_dataset.take(5)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the MRIscan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")
plt.savefig('agu_scan.png')
plt.close()

"""
Since a MRI scan has many slices, let's visualize a montage of the slices.

"""

def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 40 MRI slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
    plt.savefig('slices.png')
    plt.close()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the MRI scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])


"""
## Define a 3D convolutional neural network

To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).
"""

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

"""
## Train model
"""

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification3.h5", save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch', verbose=0
)
#early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
earlystopping = keras.callbacks.EarlyStopping(monitor ="val_acc", 
                                        mode ="min", patience = 15, 
                                        restore_best_weights = True)

# Train the model, doing validation at the end of each epoch
epochs = 100
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb,earlystopping]
)


#save the results
#model.save('model.object')

import json

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'history3.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

#load
#model = keras.models.load_model('model.object')

# Opening JSON file
history = pd.read_json('history3.json')

#model = keras.models.load_model("3d_image_classification.balanced.h5")

#save the results
#import pickle

# Store data (serialize)
#with open('model.pickle', 'wb') as handle:
#    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
#with open('filename.pickle', 'rb') as handle:
#    unserialized_data = pickle.load(handle)


#from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

hist_df = pd.DataFrame(history)

# summarize history for accuracy
plt.close()
plt.plot(hist_df.acc)
plt.plot(hist_df.val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy3.png')


# summarize history for loss
plt.close()
plt.plot(hist_df.loss)
plt.plot(hist_df.val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss3.png')


tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)


"""
## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.
"""
    
#model summary
print(model.summary())

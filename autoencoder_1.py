import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import sklearn.preprocessing as skp
import sklearn.metrics as sm

from sklearn.ensemble import RandomForestClassifier as rfc

import seaborn as sns
import lightgbm as lgbm

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras


inputs = {}
for name, column in june_df[clust_var].items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs

numeric_inputs = {name: input for name, input in inputs.items() if input.dtype == tf.float32}

x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(june_df[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]


preprocessed_inputs_cat = keras.layers.Concatenate()(preprocessed_inputs)
preprocessing_layer = tf.keras.Model(inputs, preprocessed_inputs_cat, name="ProcessData")


# this saves an image of the model, see note regarding plot_model issues
tf.keras.utils.plot_model(model=preprocessing_layer, rankdir="LR", dpi=130, show_shapes=True, to_file="processing.png")

items_features_dict = {name: np.array(value) for name, value in june_df[clust_var].items()}

# grab two samples
two_sample_dict = {name:values[1:3, ] for name, values in items_features_dict.items()}
two_sample_dict

# apply the preprocessing layer
two_sample_fitted = preprocessing_layer(two_sample_dict)
two_sample_fitted


# This is the size of our input data
full_dim = two_sample_fitted.shape.as_list()[1]

# these are the downsampling/upsampling dimensions
encoding_dim1 = 64
encoding_dim2 = 34
encoding_dim2a = 16
encoding_dim2b = 8
encoding_dim3 = 3
# we will use these 3 dimensions for clustering

# This is our encoder input
encoder_input_data = keras.Input(shape=(full_dim,))

# the encoded representation of the input
encoded_layer1 = keras.layers.Dense(encoding_dim1, activation='relu')(encoder_input_data)
encoded_layer2 = keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer1)
encoded_layer2a = keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer2)
encoded_layer2b = keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer2a)
# Note that encoded_layer3 is our 3 dimensional "clustered" layer, which we will later use for clustering
encoded_layer3 = keras.layers.Dense(encoding_dim3, activation='relu', name="ClusteringLayer")(encoded_layer2a)

encoder_model = keras.Model(encoder_input_data, encoded_layer3)

# the reconstruction of the input
decoded_layer3 = keras.layers.Dense(encoding_dim2a, activation='relu')(encoded_layer3)
decoded_layer2a = keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer3)
decoded_layer2 = keras.layers.Dense(encoding_dim1, activation='relu')(decoded_layer2a)
decoded_layer1 = keras.layers.Dense(full_dim, activation='sigmoid')(decoded_layer2)

# This model maps an input to its autoencoder reconstruction
autoencoder_model = keras.Model(encoder_input_data, outputs=decoded_layer1, name="Encoder")

# compile the model
autoencoder_model.compile(optimizer="ADAM", loss=tf.keras.losses.mean_squared_error)
# tf.keras.utils.plot_model(model=autoencoder_model, rankdir="LR", dpi=130, show_shapes=True, to_file="autoencoder.png") #RMSProp


encoder_model.summary()

# process the inputs
p_items = preprocessing_layer(items_features_dict)
p_labels = june_df['inf_flag']

# split into training and testing sets (80/20 split)
train_data, test_data, train_labels, test_labels = train_test_split(p_items.numpy(), p_labels, train_size=0.8, random_state=5)

# fit the model using the training data
history = autoencoder_model.fit(train_data, train_data, epochs=2000, batch_size=512, shuffle=True, validation_data=(test_data, test_data))

# encoder_model.save("encoder_model.h5")

encoder_model.save("encoder_save", overwrite=True)

encoder_save = tf.keras.models.load_model("encoder_save")

encoder_save = tf.keras.models.load_model("encoder_save")

encoder_save.summary()

encoded_items = encoder_save(p_items)


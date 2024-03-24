from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3 as inception
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.preprocessing.image as tf_image
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, add
import numpy as np



encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048

preprocess_input = inception.preprocess_input

def captionModel(max_length, vocab_size, embedding_dim):
    inputs1 = Input(shape=(OUTPUT_DIM,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return caption_model


def encodeImage(img):
  img = img.resize((WIDTH, HEIGHT))
  x = tf_image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  x = encode_model.predict(x) 
  x = np.reshape(x, OUTPUT_DIM )
  return x
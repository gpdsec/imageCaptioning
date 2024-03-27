import tensorrt
import tensorflow.keras.preprocessing.image as tf_image
import os
import matplotlib.pyplot as plt
from model import captionModel, encodeImage
from util import generateCaption
import pickle

#_____________________________________________________________________________________________________

WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
START = "startseq"
STOP = "endseq"
max_length = 31
vocab_size = 197
embedding_dim = 300

with open('idxtoword.pickle', "rb") as f:
   idxtoword = pickle.load(f)
with open('wordtoidx.pickle', "rb") as f:
   wordtoidx = pickle.load(f)

caption_model = captionModel(max_length, vocab_size, embedding_dim)
caption_model.load_weights('caption_model.hdf5')
image_path="Caption-Dataset/Images/bhcoradxnh.jpg"

img = tf_image.load_img(image_path, target_size=(HEIGHT, WIDTH))
img = encodeImage(img)
img = img.reshape((1,OUTPUT_DIM))
x=plt.imread(image_path)
plt.imshow(x)
plt.show()
print("Caption:",generateCaption(img, caption_model, START, STOP, max_length, wordtoidx, idxtoword))

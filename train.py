import tensorrt
from string import punctuation
import re
from nltk import word_tokenize
import nltk
import tensorflow.keras.preprocessing.image as tf_image
import os

from model import captionModel, encodeImage
from util import data_generator
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
START = "startseq"
STOP = "endseq"




# img=tf_image.load_img('images/content/images/1.jpg', target_size=(299,299))
# encodeImage(img)

data=pd.read_csv("Caption-Dataset/Images-200.csv")
data['caption']=data['caption'].apply(lambda x:START+' '+x+' '+STOP)


encoded_images={}
for i in range(len(data)):
  fname = data['file_name'][i].split("#")[0]
  image_path= os.path.join('Caption-Dataset/Images', fname)
  
  img = tf_image.load_img(image_path, target_size=(HEIGHT, WIDTH))
  encoded_images[i] = encodeImage(img)
  

data.reset_index(drop=True,inplace=True)
data['id']=[i for i in range(len(data))]


data['caption']=data['caption'].apply(lambda x:re.sub("["+punctuation+"]",' ',x))
data['caption']=data['caption'].apply(lambda x:re.sub("\d",' ',x))
data['caption']=data['caption'].apply(lambda x:re.sub("\s+",' ',x))
data['caption']=data['caption'].str.lower()




word_count_threshold = 5
word_counts = {}
for caption in data['caption']:
    for w in word_tokenize(caption):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))



caption_lens=[]
for caption in data['caption']:
  words=word_tokenize(caption)
  words=[w for w in words if w in vocab]
  caption_lens.append(len(words))
print(data['caption'])
max_length=max(caption_lens)


idxtoword = {}
wordtoidx = {}

ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1

vocab_size = len(idxtoword) + 1

max_length=max(caption_lens)

embeddings_index = {}
f = open( 'glove.42B.300d.txt', encoding="utf-8")

for line in f:
    line=line.strip()
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print(f'Found {len(embeddings_index)} word vectors.')


embedding_dim = 300

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoidx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

with open('idxtoword.pickle', 'wb') as dfile:
    pickle.dump(idxtoword, dfile)
with open('wordtoidx.pickle', 'wb') as dfile:
    pickle.dump(wordtoidx, dfile)

caption_model = captionModel(max_length, vocab_size, embedding_dim)
caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

batch_size = 3
EPOCHS = 20
steps = len(data['caption'])/batch_size


for i in tqdm(range(EPOCHS)):
    generator = data_generator(data, encoded_images, wordtoidx, max_length, batch_size, vocab_size)
    caption_model.fit(generator, epochs=3, steps_per_epoch=steps, verbose=1)

caption_model.optimizer.lr = 1e-4
number_pics_per_batch = 6
steps = len(data['caption'])//number_pics_per_batch

for i in tqdm(range(EPOCHS)):
    generator = data_generator(data, encoded_images, wordtoidx, max_length, number_pics_per_batch)
    caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

caption_model.save_weights('caption_model.hdf5')
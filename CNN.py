import tarfile
import re
import os
import glob
import numpy as np
from typing import Tuple
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors

import tensorflow as tf
from tensorflow import keras



with tarfile.open('aclImdb_v1.tar.gz', 'r') as tf:
    tf.extractall()

def get_data(filepath:str) -> list:
    pattern=re.compile('<.*?>')
    pos_path=os.path.join(filepath,'pos')
    neg_path=os.path.join(filepath,'neg')

    dataset=[]

    def append_data(label:int, path:str) -> None:
        for file in glob.glob(os.path.join(path,'*.txt')):
            with open(file,'r') as f:
                review_cleaned=re.sub(pattern,'',f.read())
                dataset.append((label,review_cleaned))

    append_data(label=1, path=pos_path)
    append_data(label=0, path=neg_path)
    np.random.shuffle(dataset)

    return dataset

dataset=get_data('aclImdb/train')


def vectorize_data(dataset:list) -> Tuple[list,list]:
    vectorized_data=[]
    target_labels=[]

    tokenizer=TreebankWordTokenizer()
    word_vectors=KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True, limit=200000
    )

    for sample in dataset:
        label, review=sample[0],sample[1]
        tokens=tokenizer.tokenize(review)
        sample_word_vec=[]

        for token in tokens:
            try:
                sample_word_vec.append(word_vectors[token])
            except KeyError:
                pass

        vectorized_data.append(sample_word_vec)
        target_labels.append(label)

    return vectorized_data,target_labels

x,y=vectorize_data(dataset)

def pad_truncate(dataset:list,maxlen:int) -> list:
    data=[]
    word_vector_size=dataset[0][0]
    zero_vector=np.zeros_like(word_vector_size)

    for sample in dataset:
        tmp=sample

        if len(sample)>maxlen:
            tmp=sample[:maxlen]

        if len(sample)<maxlen:
            add_ele=maxlen-len(sample)

            for _ in range(add_ele):
                tmp.append(zero_vector)

        data.append(tmp)

    return data

x=pad_truncate(x,maxlen=400)

def train_test_split(x:list, y:list, test_size:float, maxlen:int, embedding_dims:int) ->Tuple:
    split_point=int(len(x)*(1-test_size))

    x_train,y_train=x[:split_point],y[:split_point]
    x_test,y_test=x[split_point:],y[split_point:]

    x_train=np.reshape(x_train,(len(x_train),maxlen,embedding_dims))
    x_test=np.reshape(x_test,(len(x_test),maxlen,embedding_dims))

    y_train=np.array(y_train)
    y_test=np.array(y_test)

    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,maxlen=400,embedding_dims=300
)


inputs=keras.Input(shape=(400,300))
x= keras.layers.Conv1D(
    250,3,padding='valid',activation='relu',strides=1
)(inputs)

x=keras.layers.GlobalMaxPooling1D()(x)
x=keras.layers.Dense(250,activation='relu')(x)
x=keras.layers.Dropout(0.2)(x)

outputs=keras.layers.Dense(1,activation='sigmoid')(x)

model=keras.Model(inputs,outputs,name='cnn_imdb')

model.compile(
    loss='binary_crossentopy',
    optimizer='adam',
    metrics=['accuracy']
)

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)

hisory=model.fit(
    x_train,y_train,
    batch_size=32,
    epochs=2,
    validation_data=(x_test,y_test)
)

with open('cnn_model.json','w') as f:
    f.write(model.to_json())

model.save_weights('cnn_weights.h5')
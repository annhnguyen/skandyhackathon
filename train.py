import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist #ex. dataset
import numpy as np
import sys
import os

MODEL_PATH = 'model.h5'



'''
def build_model(input_shape, num_classes): #defining a simple model -- feedforward
    model=Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_cross entropy',
                  metrics=['accuracy'])
    return model

def train_model():
'''

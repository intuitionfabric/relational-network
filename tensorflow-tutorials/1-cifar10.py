"""
CIFAR-10 tutorial using Tensorflow Estimators and Dataset API

"""

from tensorflow.python.keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras import models
from tensorflow.python.keras import layers
import tensorflow as tf

import os

import cifar10utils
from cifar10utils import Cifar10DataSet as DataParse

cifar10utils.download_generate_cifar10_tfrecs()


conv_base = InceptionV3(weights='imagenet', include_top=True)
model = models.Sequential()
model.add(layers.UpSampling2D(size=(5,5), input_shape=(32,32,3)))
model.add(conv_base)
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))
conv_base.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


model_dir = "./models/cifar10"
os.makedirs(model_dir, exist_ok=True)
estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                    model_dir=model_dir)



def input_fn(data_dir, subset, batch_size, epochs):
    use_distortion = subset == 'train'
    dataset = DataParse(data_dir, subset, use_distortion)
    image_batch, label_batch = dataset.make_batch(batch_size, epochs)
    return {str(model.input_names[0]):image_batch}, label_batch


estimator.train(input_fn=lambda: input_fn('./cifar10-data', 'train', 64, 30))

evaluate_results = estimator.evaluate(input_fn=lambda: input_fn('./cifar10-data', 'eval', 64,1))
print("Evaluation results")
for key in evaluate_results:
    print("   {}, was: {}".format(key, evaluate_results[key]))
    

# nohup python -u -m self_driving.steering.model_saliency > self_driving/steering/output.txt 2>&1 &

from keras import applications
from keras.models import Sequential
from scipy import misc
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
import numpy as np
from keras.preprocessing.image import img_to_array
import os

VAL_DATASET = "/usr/local/google/home/limeng/Downloads/udacity/test/HMB_3/center/"

# dimensions of our images.
img_width, img_height = 224, 224
model_weights_path = 'save/steering_resnet50-22-0.0603.hdf5'

# build the resnet50 network
base_model = applications.ResNet50(include_top=False,
                                   input_shape=(224, 224, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(64, activation='relu'))
top_model.add(Dense(1))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.load_weights(model_weights_path)

with open("output/steering/steering_val.txt", 'a') as out:
    for img in os.listdir(VAL_DATASET):
        img_data = utils.load_img(VAL_DATASET + img, target_size=(224, 224))
        img_input = np.expand_dims(img_to_array(img_data), axis=0)
        out.write("%s %.10f\n" % (img, model.predict(img_input / 255.)[0][0]))
        out.flush()
        heat_map = visualize_saliency(model,
                                      -2,
                                      filter_indices=None,
                                      seed_input=img_data,
                                      backprop_modifier='guided')
        misc.imsave("output/steering/%s" % img, overlay(img_data, heat_map, alpha=0.3))

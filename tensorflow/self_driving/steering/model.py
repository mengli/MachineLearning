from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from utils import nvida


# dimensions of our images.
img_width, img_height = 455, 256

epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

(x_train, y_train), (x_test, y_test) = nvida.load_data()

model = Sequential()
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=input_shape))
model.add(Activation('relu'))

model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
image_datagen = ImageDataGenerator(rescale=1. / 255)

# fits the model on batches with real-time data augmentation:
model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) // batch_size, epochs=epochs,
                    validation_data=image_datagen.flow(x_test, y_test, batch_size=batch_size),
                    validation_steps=len(x_test) // batch_size)

model.save_weights('first_try.h5')

# nohup python -u -m self_driving.steering.model_resnet50 > self_driving/steering/output.txt 2>&1 &

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from utils import my_image
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'utils/udacity_train.txt'
validation_data_dir = 'utils/udacity_val.txt'
nb_train_samples = 33808
nb_validation_samples = 10558
epochs = 50
batch_size = 32

# build the resnet50 network
base_model = applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(64, activation='relu'))
top_model.add(Dense(1))

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# compile the model with a Adam optimizer
# and a very slow learning rate.
model.compile(loss=root_mean_squared_error,
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = my_image.MyImageDataGenerator(rescale=1. / 255)

test_datagen = my_image.MyImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(
    train_data_dir,
    [img_width, img_height, 3],
    shuffle=True)

validation_generator = test_datagen.flow(
    validation_data_dir,
    [img_width, img_height, 3],
    shuffle=True)

# checkpoint
filepath="save/steering_resnet50-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='min')
callbacks_list = [checkpoint]

model.summary()

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

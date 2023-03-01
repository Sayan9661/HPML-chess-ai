from tensorflow import keras
import logging
import tensorflow.keras.callbacks as callbacks
import os
import numpy
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import argparse

# import logging
# logging.basicConfig(filename='trymodels.log', encoding='utf-8',level=logging.INFO)



COLOR = 'white'

NUM_LAYERS = 5#3 to 8
BATCH_SIZE = 2048 #vary between 512 t0 4096
conv_size= 64# 32 64  128 256
REGULARIZATION = False # true, false
MODEL_TYPE = 'conv' # conv res

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers")
parser.add_argument("--conv_size")

parser.add_argument("--model_type")
parser.add_argument("--batch_size")
args = parser.parse_args()


conv_size=int(args.conv_size)
NUM_LAYERS=int(args.num_layers)

BATCH_SIZE=int(args.batch_size)


MODEL_TYPE=(args.model_type)

def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(8, 8, 8))

    # adding the convolutional layers
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same',
                          activation='relu', data_format='channels_first')(x)
        x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    return models.Model(inputs=board3d, outputs=x)


"""Skip connections (residual network) will likely improve the model for deeper connections. If you want to test the residual model, check the code below."""


def build_model_residual(conv_size, conv_depth):
  board3d = layers.Input(shape=(8, 8, 8))

  # adding the convolutional layers
  x = layers.Conv2D(filters=conv_size, kernel_size=3,
                    padding='same', data_format='channels_first')(board3d)
  for _ in range(conv_depth):
    previous = x
    x = layers.Conv2D(filters=conv_size, kernel_size=3,
                      padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=conv_size, kernel_size=3,
                      padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, previous])
    x = layers.Activation('relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(1, 'sigmoid')(x)

  return models.Model(inputs=board3d, outputs=x)


def get_dataset():
  return numpy.load(f'{COLOR}_large_np.npy'), numpy.load(f'{COLOR}_large_score.npy')


x_train, y_train = get_dataset()

if(MODEL_TYPE =='conv'):
    model = build_model(conv_size, NUM_LAYERS)
else:
    model = build_model_residual(conv_size, NUM_LAYERS)


model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
model.summary()
model.fit(x_train, y_train,
          BATCH_SIZE,
          epochs=1000,
          verbose=2,
          validation_split=0.1,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                     callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])


# val_loss = 
# logging.info(f'model_{MODEL_TYPE}_{NUM_LAYERS}_{conv_size}_{str(REGULARIZATION)}',val_loss)

model.save(f'model_{MODEL_TYPE}_{NUM_LAYERS}_{conv_size}_{str(REGULARIZATION)}.h5')
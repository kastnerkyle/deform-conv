from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from deform_conv.layers import ConvOffset2D
from deform_conv.callbacks import TensorBoard
from deform_conv.cnn import get_cnn, get_deform_cnn
from deform_conv.mnist import get_gen
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# ---
# Config

batch_size = 32
n_train = 60000
n_test = 10000
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False
)


# ---
# Normal CNN

inputs, outputs = get_cnn()
model = Model(inputs=inputs, outputs=outputs)
model.summary()
optim = Adam(1e-3)
# optim = SGD(1e-3, momentum=0.99, nesterov=True)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])

model.fit_generator(
    train_gen, steps_per_epoch=steps_per_epoch,
    epochs=10, verbose=1,
    validation_data=test_gen, validation_steps=validation_steps
)
model.save_weights('models/cnn.h5')
# 1875/1875 [==============================] - 24s - loss: 0.0090 - acc: 0.9969 - val_loss: 0.0528 - val_acc: 0.9858

# ---
# Evaluate normal CNN

model.load_weights('models/cnn.h5', by_name=True)

val_loss, val_acc = model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy', val_acc)
# 0.9874

val_loss, val_acc = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy with scaled images', val_acc)
# 0.5701

# ---
# Deformable CNN

inputs, outputs = get_deform_cnn(trainable=False)
model = Model(inputs=inputs, outputs=outputs)
model.load_weights('models/cnn.h5', by_name=True)
model.summary()
optim = Adam(5e-4)
# optim = SGD(1e-4, momentum=0.99, nesterov=True)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])

model.fit_generator(
    train_scaled_gen, steps_per_epoch=steps_per_epoch,
    epochs=20, verbose=1,
    validation_data=test_scaled_gen, validation_steps=validation_steps
)
# Epoch 20/20
# 1875/1875 [==============================] - 504s - loss: 0.2838 - acc: 0.9122 - val_loss: 0.2359 - val_acc: 0.9231
model.save_weights('models/deform_cnn.h5')

# --
# Evaluate deformable CNN

model.load_weights('models/deform_cnn.h5')

val_loss, val_acc = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with scaled images', val_acc)
# 0.9255

val_loss, val_acc = model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with regular images', val_acc)
# 0.9727

deform_conv_layers = [l for l in model.layers if isinstance(l, ConvOffset2D)]

Xb, Yb = next(test_gen)
for l in deform_conv_layers:
    print(l)
    _model = Model(inputs=inputs, outputs=l.output)
    offsets = _model.predict(Xb)
    offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
    print(offsets.min())
    print(offsets.mean())
    print(offsets.max())

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations
from pinn_tbl import pinns
from utils import get_data

from time import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

fname = '../data/data_ub1.mat'

bc, cp = get_data(fname)

nn = 20
nl = 4
act = activations.tanh
inp = layers.Input(shape = (2,))
hl = inp
for i in range(nl):
    hl = layers.Dense(nn, activation = act)(hl)
out = layers.Dense(5)(hl)

model = models.Model(inp, out)
print(model.summary())

initial_learning_rate = 1e-2
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=5000,
    decay_rate=0.1,
    staircase=True)

opt = optimizers.Adam(initial_learning_rate)
st_time = time()

pinn = pinns(model, opt, 20000)
hist = pinn.fit(bc, cp)

en_time = time()
comp_time = en_time - st_time



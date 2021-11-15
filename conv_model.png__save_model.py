import numpy as np
import os
from tensorflow import keras
os.chdir(r'E:\0mProbe\Projects\proj_20210818_captcha_recognition_r_python')
model = keras.models.load_model('conv_model.png.model')
keras.utils.plot_model(
  model = model,
  to_file = 'conv_model.png',
  show_shapes = True,
  show_layer_names = False
)

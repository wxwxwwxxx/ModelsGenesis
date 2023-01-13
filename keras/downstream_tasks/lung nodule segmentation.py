#!/usr/bin/env python
# coding: utf-8

# # Processed LIDC data can be found at: https://drive.google.com/drive/folders/1TLpPvR_9hfNdUbD9dFIXNpJ7m50VmD19?usp=sharing

# In[1]:


#get_ipython().system('pip install -r requirements.txt')


# In[1]:


#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import keras
print("keras = {}".format(keras.__version__))
import tensorflow as tf
print("tensorflow-gpu = {}".format(tf.__version__))
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass
import random
import shutil
import argparse
import sklearn
from pathlib import Path
from utils import *
from unet3d import *
from config import *
from ncs_data import *

class set_args():
    gpu = 0
    data = None
    apps = 'ncs'
    run = 1
    cv = None
    subsetting = None
    suffix = 'genesis'
    task = 'segmentation'
    
args = set_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    

conf = ncs_config(args)


# In[2]:


x_train, y_train = load_image(conf, 'train')
print('x_train: {} | {} ~ {}'.format(x_train.shape, np.min(x_train), np.max(x_train)))
print('y_train: {} | {} ~ {}'.format(y_train.shape, np.min(y_train), np.max(y_train)))

x_valid, y_valid = load_image(conf, 'valid')
print('x_valid: {} | {} ~ {}'.format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
print('y_valid: {} | {} ~ {}'.format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

x_test, y_test = load_image(conf, 'test')
print('x_test: {} | {} ~ {}'.format(x_test.shape, np.min(x_test), np.max(x_test)))
print('y_test: {} | {} ~ {}'.format(y_test.shape, np.min(y_test), np.max(y_test)))


# # Fine-tune Models Genesis

# In[3]:


args.suffix = 'genesis'
conf = ncs_config(args)
conf.display()


# ### Train

# In[ ]:


model = unet_model_3d((1,conf.input_rows,conf.input_cols,conf.input_deps), batch_normalization=True)
if conf.weights is not None:
    print("[INFO] Load pre-trained weights from {}".format(conf.weights))
    model.load_weights(conf.weights)
model, callbacks = model_setup(model, conf, task=args.task)

while conf.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit(x_train, y_train,
                  validation_data=(x_valid, y_valid),
                  batch_size=conf.batch_size,
                  epochs=conf.nb_epoch, 
                  verbose=conf.verbose, 
                  shuffle=True,
                  callbacks=callbacks)
        break
    except tf.errors.ResourceExhaustedError as e:
        conf.batch_size = int(conf.batch_size - 2)
        print("\n> Batch size = {}".format(conf.batch_size))


# ### Test

# In[11]:


model = unet_model_3d((1,conf.input_rows,conf.input_cols,conf.input_deps), batch_normalization=True)
print("[INFO] Load trained model from {}".format( os.path.join(conf.model_path, conf.exp_name+".h5") ))
model.load_weights(os.path.join(conf.model_path, conf.exp_name+".h5") )

p_test = segmentation_model_evaluation(model=model, config=conf, x=x_test, y=y_test, note=conf.exp_name)


# ### Visualization

# In[12]:


p_test = np.squeeze(p_test)
for i in range(0, x_test.shape[0], 80):
    plot_image_truth_prediction(x_test[i], y_test[i], p_test[i], rows=5, cols=5)


# # Train from scratch

# In[3]:


args.suffix = 'random'
conf = ncs_config(args)
conf.display()


# ### Train

# In[4]:


model = unet_model_3d((1,conf.input_rows,conf.input_cols,conf.input_deps), batch_normalization=True)
if conf.weights is not None:
    print("[INFO] Load pre-trained weights from {}".format(conf.weights))
    model.load_weights(conf.weights)
model, callbacks = model_setup(model, conf, task=args.task)

while conf.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit(x_train, y_train,
                  validation_data=(x_valid, y_valid),
                  batch_size=conf.batch_size,
                  epochs=conf.nb_epoch, 
                  verbose=conf.verbose, 
                  shuffle=True,
                  callbacks=callbacks)
        break
    except tf.errors.ResourceExhaustedError as e:
        conf.batch_size = int(conf.batch_size - 2)
        print("\n> Batch size = {}".format(conf.batch_size))


# ### Test

# In[4]:


model = unet_model_3d((1,conf.input_rows,conf.input_cols,conf.input_deps), batch_normalization=True)
print("[INFO] Load trained model from {}".format( os.path.join(conf.model_path, conf.exp_name+".h5") ))
model.load_weights( os.path.join(conf.model_path, conf.exp_name+".h5") )

p_test = segmentation_model_evaluation(model=model, config=conf, x=x_test, y=y_test, note=conf.exp_name)


# ### Visualization

# In[5]:


p_test = np.squeeze(p_test)
for i in range(0, x_test.shape[0], 80):
    plot_image_truth_prediction(x_test[i], y_test[i], p_test[i], rows=5, cols=5)


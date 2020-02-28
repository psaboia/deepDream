# Auxiliary functions

import scipy
import imageio
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing import image
from keras import backend as k

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    # scipy.misc.imsave(fname, pil_img) # scipy.misc.imsave has been deprecated in newer Scipy versions.
    imageio.imwrite(fname, pil_img)
    
# Util function to open, resize, and format pictures into tensors that Inception V3 can process    
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

# Util function to convert a tensor into a valid image
def deprocess_image(x):
    if k.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        # Undoes preprocessing that was performed by inception_v3.preprocess_input
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

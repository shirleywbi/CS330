import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import utils
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

from skimage.segmentation import mark_boundaries

import lime
from lime import lime_image

resnet = ResNet50(weights='imagenet')

img_bowtie = load_img("image_path", target_size=(224, 224))
plt.imshow(img_bowtie)
x_bowtie = img_to_array(img_bowtie).astype('double')
x_bowtie = np.expand_dims(x_bowtie, axis=0)
x_bowtie = preprocess_input(x_bowtie)
preds_bowtie = resnet.predict(x_bowtie)
decode_predictions(preds_bowtie, top=3)[0] # returns predictions of what it thinks the image is

# Adapted from LIME example notebook
explainer_bowtie = lime_image.LimeImageExplainer() 
explanation_bowtie = explainer_bowtie.explain_instance(x_bowtie[0], resnet.predict,
                                                       top_labels=5, hide_color=0, num_samples=1000)
temp_bowtie, mask_bowtie = explanation_bowtie.get_image_and_mask(explanation_bowtie.top_labels[0], 
                                                                 positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp_bowtie, mask_bowtie))
from RPN import RPN
from utils import read_data, plot_image
import tensorflow as tf
import numpy as np

# read data
PATH = "anotations"
bbox, images = read_data(PATH)

bbox_one = tf.expand_dims(bbox[0], axis = 0)
images_one = tf.expand_dims(images[0], axis = 0)
# get backbone

vvg16 = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# get feature map of the images
res = vvg16.predict(images)

# train RPN model

model = RPN(9  , range_positive=0.5,
                 range_negative=0.1,
                 scales=np.array([0.5, 1, 2]),
                 dims=np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))
model.build((None, 20, 20, 512))
model.compile(
    optimizer=tf.keras.optimizers.Adam()
)

history = model.fit(res, bbox, batch_size=1, epochs=5000)

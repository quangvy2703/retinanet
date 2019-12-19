# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf



# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
# keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
print(model_path)

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = {0: "dashline"}
THRES_SCORE = 0.8


def img_inference(img_path):
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    print("Preshape ", image.shape)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    print("Shape ", image.shape)
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


images = os.listdir("/media/vy/DATA/data/dashline/images/test2017")

for image in images:
    print('Running inference on: ' + "/media/vy/DATA/data/dashline/images/test2017/" + image)
    img_inference("/media/vy/DATA/data/dashline/images/test2017/" + image)
    input("Press Enter to continue...")

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import ImageDraw
from matplotlib import pyplot as plt



# def draw_bboxes_with_labels(element, probs, labels):
#     img = element['image']
#     bboxes = element['objects']['bbox']
#     label_indices = element['objects']['label']
#     image = tf.keras.preprocessing.image.array_to_img(img)
#     width, height = image.size
#     bboxes = denormalize_bboxes(bboxes, width, height)
#     colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
#     draw = ImageDraw.Draw(image)
#     for index, bbox in enumerate(bboxes):
#         y1, x1, y2, x2 = tf.split(bbox, 4)
#         width = x2 - x1
#         height = y2 - y1
#         if width <= 0 or height <= 0:
#             continue
#         label_index = int(label_indices[index])
#         color = tuple(colors[label_index].numpy())
#         label_text = "{0} {1:0.3f}".format(labels[label_index], 1)
#         draw.text((x1 + 4, y1 + 2), label_text, fill=color)
#         draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
#     #
#     plt.figure()
#     plt.imshow(image)
#     plt.show()


def denormalize_bboxes(bboxes, height, width):
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


def draw_bboxes_with_labels(img, bboxes, label_indices, probs, labels):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in denormalized form
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
    bboxes = denormalize_bboxes(bboxes, height, width)
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    draw = ImageDraw.Draw(image)
    for index, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label_index = int(label_indices[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    #
    plt.figure()
    plt.imshow(image)
    plt.show()

train_data = tfds.load("coco/2017", split='validation')

for element in train_data:
    if element['image/id'] != 397133:
        continue
    img = element['image']
    bboxes = element['objects']['bbox']


    draw_bboxes_with_labels(img, bboxes, element['objects']['label'], list(range(100)), list(range(100)))

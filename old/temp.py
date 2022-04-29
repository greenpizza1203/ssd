import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
from PIL import Image, ImageDraw
def denormalize_bboxes(bboxes, height, width):
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


def draw_bboxes_with_labels(img, bboxes, label_indices, labels):
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
        label_text = "{0}".format(labels[label_index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    #
    plt.figure()
    plt.imshow(image)
    plt.show()

element = next(iter(val_data))
colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
for element in val_data.take(10):
    img = element['image']
    bboxes = element['objects']['bbox']
    draw_bboxes_with_labels(img, bboxes, element['objects']['label'], labels)
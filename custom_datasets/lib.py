import tensorflow as tf
import tensorflow_datasets as tfds

image_size = (300, 300)


def resize_image(element):
    image = tf.image.convert_image_dtype(element['image'], tf.float32)
    resized_image = tf.image.resize(image, image_size)
    fixed_image = tf.image.convert_image_dtype(resized_image, tf.uint8)
    element['image'] = fixed_image
    return element


def load_dataset(name, split):
    data, info = tfds.load(name, split=split, with_info=True, download=False)

    labels = info.features["labels"].names

    total = 0
    for part in split.split("+"):
        total += info.splits[part].num_examples

    return data, total, labels


class Model:
    def __init__(self, train=None, val=None, test=None, labels=None):
        self.train = train
        self.val = val
        self.test = test
        self.labels = labels


class Split:
    def __init__(self, data, size):
        self.data = data
        self.size = size

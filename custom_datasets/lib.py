import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

image_size = (300, 300)


def resize_image(element):
    image = tf.image.convert_image_dtype(element['image'], tf.float32)
    resized_image = tf.image.resize(image, image_size)
    fixed_image = tf.image.convert_image_dtype(resized_image, tf.uint8)
    element['image'] = fixed_image
    return element


def load_dataset(name, split):
    data, info = tfds.load(name, split=split, with_info=True)

    labels = info.features["labels"].names

    total = 0
    for part in split.split("+"):
        total += info.splits[part].num_examples

    return data, total, labels


class Split:
    def __init__(self, data, size):
        self.data = data
        self.size = size

    def cache(self, path):
        os.makedirs(path)
        computed_size = 0
        self.data = self.data.cache(path)
        for _ in tqdm(self.data, total=self.size):
            computed_size += 1
        assert computed_size == self.size

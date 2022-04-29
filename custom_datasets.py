import tensorflow as tf
import tensorflow_datasets as tfds

image_size = (300, 300)


def load_dataset(name, split):
    data, info = tfds.load(name, split=split, with_info=True)

    labels = info.features["labels"].names

    total = 0
    for part in split.split("+"):
        total += info.splits[part].num_examples

    return data, total, labels


def resize_image(element):
    image = tf.image.convert_image_dtype(element['image'], tf.float32)
    resized_image = tf.image.resize(image, image_size)
    fixed_image = tf.image.convert_image_dtype(resized_image, tf.uint8)
    return fixed_image


class Voc:
    def __init__(self):
        train_data, train_size, labels = load_dataset("voc/2007", "train+validation")
        train_2012, train_2012_size, _ = load_dataset("voc/2012", "train+validation")

        train_data = train_data.concatenate(train_2012).map(resize_image)

        val_data, val_size, _ = load_dataset("voc/2007", "test")

        val_data = val_data.map(resize_image)
        self.train_data = train_data
        self.train_size = train_size + train_2012_size
        self.val_data = val_data
        self.val_size = val_size
        self.labels = labels

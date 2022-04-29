import tensorflow as tf
import tensorflow_datasets as tfds

image_size = (300, 300)


def resize_image(element):
    image = tf.image.convert_image_dtype(element['image'], tf.float32)
    resized_image = tf.image.resize(image, image_size)
    fixed_image = tf.image.convert_image_dtype(resized_image, tf.uint8)
    return fixed_image


train_data, info = tfds.load("voc/2007", split="train+validation", with_info=True)

voc_2012_data, voc_2012_info = tfds.load("voc/2012", split="train+validation", with_info=True)

voc_2012_total_items = voc_2012_info.splits["train"].num_examples + voc_2012_info.splits["validation"].num_examples

train_data = train_data.concatenate(voc_2012_data).map(resize_image)

val_data = tfds.load("voc/2007", split="test").map(resize_image)

labels = info.features["labels"].names

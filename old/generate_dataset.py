import json
import os
import shutil

import tensorflow as tf
import tensorflow_datasets as tfds

datasets, info = tfds.load('coco/2017', with_info=True)

label_names = info.features['objects']['label'].names
person, car = label_names.index('person'), label_names.index('car')

matched_labels = tf.constant([person, car], tf.int64)


def filter_label(img):
    labels = img['objects']['label']
    intersect = tf.sets.intersection([matched_labels], [labels])
    num_matches = tf.size(intersect)

    return num_matches > 0


img_size = 300


def trim_element(element):
    label = element['objects']['label']
    ind = tf.where(tf.logical_or(label == 0, label == 2))
    bbox = element['objects']['bbox']

    image = tf.image.convert_image_dtype(element['image'], tf.float32)
    resized_image = tf.image.resize(image, (img_size, img_size))
    fixed_image = tf.image.convert_image_dtype(resized_image, tf.uint8)
    return {
        'image': fixed_image,
        'image/id': element['image/id'],
        'objects': {
            'bbox': tf.gather_nd(bbox, ind),
            'label': tf.gather_nd(label, ind) // 2,
        }
    }


train_trimmed = datasets['train'].filter(filter_label).map(trim_element)
val_trimmed = datasets['validation'].filter(filter_label).map(trim_element)

# save_dir = 'coco-ssd'

# single = next(iter(datasets['validation'].skip(2).take(1)))
#
# if os.path.exists(save_dir):
#     shutil.rmtree(save_dir)
#
#
# def shard_function(image):
#     return tf.math.floormod(image['image/id'], 16)
#
# tf.data.experimental.save(val_trimmed, f'{save_dir}/validation', shard_func=None, compression='GZIP')
# # tf.data.experimental.save(train_trimmed, f'{save_dir}/train', shard_func=shard_function, compression='GZIP')
#
# data = {
#     # 'train_size': int(train_trimmed.reduce(0, lambda x, _: x + 1)),
#     'validation_size': int(val_trimmed.reduce(0, lambda x, _: x + 1)),
#     'labels': ['person', 'car']
# }
# with open(f'{save_dir}/data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

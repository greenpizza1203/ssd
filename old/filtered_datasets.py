from utils import data_utils
import tensorflow as tf
# def trim_element(element):
#     label = element['objects']['label']
#     ind = tf.where(tf.logical_or(label == 0, label == 2))
#     bbox = element['objects']['bbox']
#
#     image = tf.image.convert_image_dtype(element['image'], tf.float32)
#     resized_image = tf.image.resize(image, (img_size, img_size))
#     fixed_image = tf.image.convert_image_dtype(resized_image, tf.uint8)
#     return {
#         'image': fixed_image,
#         'image/id': element['image/id'],
#         'objects': {
#             'bbox': tf.gather_nd(bbox, ind),
#             'label': tf.gather_nd(label, ind) // 2,
#         }
#     }
# def filter()
test_data, info = data_utils.get_dataset("voc/2007", "test")
total_items = data_utils.get_total_item_size(info, "test")

print(next(iter(test_data)))
# print(info.features['objects']['label'].names)



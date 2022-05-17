from .lib import Split, resize_image, load_dataset, dataset_size
import tensorflow as tf

image_size = (300, 300)


# def load_datasets(labels):


# print(label_ids)



# train_data = train_data.map(filter_features).filter(has_labels)
# val_data = val_data.map(filter_features).filter(has_labels)


class Voc:
    train = val = test = None

    def __init__(self, labels):
        self.labels = labels

    @staticmethod
    def download_data():
        load_dataset("voc/2007")
        load_dataset("voc/2012")

    def load_data(self):
        train_data, all_labels = load_dataset("voc/2007", "train+validation")
        label_ids = tf.constant(list(map(lambda label: all_labels.index(label), self.labels)), dtype=tf.int64)
        train_2012, _ = load_dataset("voc/2012", "train+validation")
        train_data = train_data.concatenate(train_2012)

        val_data, _ = load_dataset("voc/2007", "test")

        test_data, _ = load_dataset("voc/2007", "test")

        def has_labels(element):
            orig_labels = element['objects']['label']
            a0 = tf.expand_dims(orig_labels, 1)
            b0 = tf.expand_dims(label_ids, 0)
            return tf.reduce_any(a0 == b0)

        train_data = train_data.filter(has_labels)
        val_data = val_data.filter(has_labels)
        test_data = test_data.filter(has_labels)

        train_size = dataset_size(train_data)
        val_size = dataset_size(val_data)
        test_size = dataset_size(test_data)
        label_ids = tf.cast(label_ids, dtype=tf.int32)

        def remove_features(element):
            orig_bboxes = element['objects']['bbox']
            orig_labels = tf.cast(element['objects']['label'], dtype=tf.int32)
            a0 = tf.expand_dims(orig_labels, 1)
            b0 = tf.expand_dims(label_ids, 0)
            indices = tf.where(tf.reduce_any(a0 == b0, 1))
            # tf.print(tf.gather_nd(orig_labels, indices))
            return {
                'objects': {
                    'bbox_old': orig_bboxes,
                    'label_old': orig_labels,
                    'bbox': tf.gather_nd(orig_bboxes, indices),
                    'label': tf.gather_nd(orig_labels, indices)
                }
            }
        train_data = train_data.map(remove_features)
        val_data = val_data.map(remove_features)
        test_data = test_data.map(remove_features)

        self.train = Split(train_data, train_size)
        self.val = Split(val_data, val_size)
        self.test = Split(test_data, test_size)

    def cache_splits(self, cache_path):
        self.train.cache(cache_path + 'train')
        self.val.cache(cache_path + 'val')
        self.test.cache(cache_path + 'test')

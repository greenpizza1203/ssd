import tensorflow_datasets as tfds

import kaggle_coco  # Register MyDataset

ds = tfds.load('kaggle_coco')
print(ds)
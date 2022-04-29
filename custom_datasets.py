import tensorflow_datasets as tfds

train_data, info = tfds.load("voc/2007", split="train+validation", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", with_info=True)

train_total_items = info.splits["train"].num_examples + info.splits["validation"].num_examples
val_total_items = info.splits["test"].num_examples

voc_2012_data, voc_2012_info = tfds.load("voc/2012", split="train+validation", with_info=True)

voc_2012_total_items = voc_2012_info.splits["train"].num_examples + voc_2012_info.splits["validation"].num_examples

train_total_items += voc_2012_total_items
train_data = train_data.concatenate(voc_2012_data)

labels = info.features["labels"].names
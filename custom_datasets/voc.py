from .lib import Split, Model, resize_image, load_dataset

image_size = (300, 300)

train_data, train_size, labels = load_dataset("voc/2007", "train+validation")
train_2012, train_2012_size, _ = load_dataset("voc/2012", "train+validation")
train_data = train_data.concatenate(train_2012).map(resize_image)
val_data, val_size, _ = load_dataset("voc/2007", "test")
val_data = val_data.map(resize_image)

voc = Model(
    train=Split(train_data, train_size),
    val=Split(val_data, val_size),
    labels=labels
)

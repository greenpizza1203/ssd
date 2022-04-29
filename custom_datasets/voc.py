from .lib import Split, resize_image, load_dataset

train = val = labels = None


def load_datasets():
    global train, val, labels
    train_data, train_size, labels = load_dataset("voc/2007", "train+validation")
    train_2012, train_2012_size, _ = load_dataset("voc/2012", "train+validation")
    train_data = train_data.concatenate(train_2012).map(resize_image)
    train_size += train_2012_size

    val_data, val_size, _ = load_dataset("voc/2007", "test")
    val_data = val_data.map(resize_image)

    train = Split(train_data, train_size)
    val = Split(val_data, val_size)
    labels = labels

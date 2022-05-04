from .lib import Split, resize_image, load_dataset

train = val = test = labels = None

image_size = (300, 300)


def filter_features(element):
    return {
        'image': resize_image(element['image'], image_size),
        'objects': {
            'bbox': element['objects']['bbox'],
            'label': element['objects']['label']
        }
    }


def load_datasets():
    global train, val, test, labels
    train_data, train_size, labels = load_dataset("voc/2007", "train+validation")
    train_2012, train_2012_size, _ = load_dataset("voc/2012", "train+validation")
    train_data = train_data.concatenate(train_2012).map(filter_features)
    train_size += train_2012_size
    train = Split(train_data, train_size)

    val_data, val_size, _ = load_dataset("voc/2007", "test")
    val_data = val_data.map(filter_features)
    val = Split(val_data, val_size)
    test_data, test_size, _ = load_dataset("voc/2007", "test")

    test = Split(test_data, test_size)
    labels = labels

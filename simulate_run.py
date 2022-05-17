from evaluator import Evaluator
from trainer import Trainer

batch_size = 16
epochs = 1

from custom_datasets import Voc

labels = ["person", "car"]
voc = Voc(labels)

voc.load_data()

print(voc.train.size)
print(voc.val.size)
print(voc.test.size)
data_point = next(iter(voc.train.data.skip(2)))
voc.download_data()

# tf.print(data_point)
# plt.imshow(data_point['image'])
# plt.show()
# print(data_point['objects'])
# train_data, train_size_estimate, val_data, val_size_estimate = voc.train_data, voc.train_size, voc.val_data, voc.val_size

trainer = Trainer(epochs, batch_size)
# trainer.load_data(voc)
# trainer.fit("output/temp.h5")

evaluator = Evaluator(batch_size)
# evaluator.load_data(voc)
# evaluator.evaluate("output/voc_150epochs_64batches.h5")

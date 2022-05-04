from evaluator import Evaluator
from trainer import Trainer
from custom_datasets import voc

batch_size = 16
epochs = 1

from custom_datasets import voc

voc.load_datasets()

# print(next(iter(voc.train.data)))
# train_data, train_size_estimate, val_data, val_size_estimate = voc.train_data, voc.train_size, voc.val_data, voc.val_size

trainer = Trainer(epochs, batch_size)
trainer.load_data(voc)
trainer.fit("output/temp.h5")

# evaluator = Evaluator(batch_size)
# evaluator.load_data(voc)
# evaluator.evaluate("output/voc_150epochs_64batches.h5")

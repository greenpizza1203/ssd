from trainer import Trainer

batch_size = 16
epochs = 1

from custom_datasets import Voc

voc = Voc()
train_data, train_size_estimate, val_data, val_size_estimate = voc.train_data, voc.train_size, voc.val_data, voc.val_size

trainer = Trainer(epochs, batch_size)
trainer.load_data(train_data, val_data, labels)
trainer.fit()

from trainer import Trainer

batch_size = 16
epochs = 1

from custom_datasets import train_data, val_data, labels

trainer = Trainer(epochs, batch_size)
trainer.load_data(train_data, val_data, labels)
trainer.fit()

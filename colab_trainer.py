from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

import augmentation
from ssd_loss import CustomLoss
from utils import bbox_utils, data_utils, io_utils, train_utils


class ColabModel:
    def __init__(self, batch_size=32, epochs=150, learning_rate=1e-3):
        self.fit_lambda = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.backbone = "mobilenet_v2"
        self.learning_rate = learning_rate

    def load_data(self, train_data, val_data, labels):
        if self.backbone == "mobilenet_v2":
            from models.ssd_mobilenet_v2 import get_model, init_model
        else:
            from models.ssd_vgg16 import get_model, init_model
        hyper_params = train_utils.get_hyper_params(self.backbone)

        train_total_items = train_data.reduce(0, lambda x, _: x + 1)
        val_total_items = val_data.reduce(0, lambda x, _: x + 1)
        # train_data = data_utils.get_dataset("voc/2007", "train+validation")
        # val_data, _ = data_utils.get_dataset("voc/2007", "test")
        # train_total_items = data_utils.get_total_item_size(info, "train+validation")
        # val_total_items = data_utils.get_total_item_size(info, "test")

        # labels = data_utils.get_labels(info)
        labels = ["bg"] + labels
        hyper_params["total_labels"] = len(labels)
        img_size = hyper_params["img_size"]
        train_data = train_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size, augmentation.apply))
        val_data = val_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size))
        data_shapes = data_utils.get_data_shapes()
        padding_values = data_utils.get_padding_values()
        train_data = train_data.shuffle(self.batch_size * 4).padded_batch(self.batch_size, padded_shapes=data_shapes,
                                                                          padding_values=padding_values)
        val_data = val_data.padded_batch(self.batch_size, padded_shapes=data_shapes, padding_values=padding_values)

        ssd_model = get_model(hyper_params)
        ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
        ssd_model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                          loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
        init_model(ssd_model)
        #
        ssd_model_path = io_utils.get_model_path(self.backbone)
        # if load_weights:
        #     ssd_model.load_weights(ssd_model_path)
        ssd_log_path = io_utils.get_log_path(self.backbone)
        # We calculate prior boxes for one time and use it for all
        # operations because of the all images are the same sizes
        prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
        ssd_train_feed = train_utils.generator(train_data, prior_boxes, hyper_params)
        ssd_val_feed = train_utils.generator(val_data, prior_boxes, hyper_params)
        checkpoint_callback = ModelCheckpoint(ssd_model_path, monitor="val_loss", save_best_only=True,
                                              save_weights_only=True)
        tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
        learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)
        step_size_train = train_utils.get_step_size(train_total_items, self.batch_size)
        step_size_val = train_utils.get_step_size(val_total_items, self.batch_size)
        self.fit_lambda = lambda: ssd_model.fit(ssd_train_feed,
                                                steps_per_epoch=step_size_train,
                                                validation_data=ssd_val_feed,
                                                validation_steps=step_size_val,
                                                epochs=self.epochs,
                                                callbacks=[checkpoint_callback, tensorboard_callback,
                                                           learning_rate_callback])

    def fit(self):
        return self.fit_lambda()

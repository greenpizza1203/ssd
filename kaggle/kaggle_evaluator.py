import tensorflow as tf
from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils, eval_utils
from models.decoder import get_decoder_model
from models.ssd_mobilenet_v2 import get_model, init_model


class KaggleEvaluator:
    def __init__(self, batch_size=32):
        self.fit_lambda = None
        self.batch_size = batch_size
        self.backbone = "mobilenet_v2"

    def evaluate(self, test_data, total_items, labels):
        backbone = self.backbone
        batch_size = self.batch_size
        io_utils.is_valid_backbone(backbone)

        #
        hyper_params = train_utils.get_hyper_params(backbone)
        #
        # test_data, info = data_utils.get_dataset("voc/2007", "test")
        # total_items = data_utils.get_total_item_size(info, "test")
        # labels = data_utils.get_labels(info)
        labels = ["bg"] + labels
        hyper_params["total_labels"] = len(labels)
        img_size = hyper_params["img_size"]

        data_shapes = data_utils.get_data_shapes()
        padding_values = data_utils.get_padding_values()

        test_data = test_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size, evaluate=False))

        test_data = test_data.padded_batch(self.batch_size, padded_shapes=data_shapes, padding_values=padding_values)

        ssd_model = get_model(hyper_params)
        ssd_model_path = io_utils.get_model_path(backbone)
        ssd_model.load_weights(ssd_model_path)

        prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
        ssd_decoder_model = get_decoder_model(ssd_model, prior_boxes, hyper_params)

        step_size = train_utils.get_step_size(total_items, batch_size)
        pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(test_data, steps=step_size, verbose=1)

        eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
        # drawing_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)

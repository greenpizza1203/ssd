from models.decoder import get_decoder_model
from utils import bbox_utils, data_utils, train_utils, eval_utils
from models.ssd_mobilenet_v2 import get_model


class Evaluator:
    def __init__(self, batch_size):
        self.evaluator = None
        self.batch_size = batch_size

    def load_data(self, dataset):
        test_data, total_items, labels = dataset.test.data, dataset.test.size, dataset.labels

        test_data = dataset.test.data
        total_items = dataset.test.size

        hyper_params = train_utils.get_hyper_params()
        #
        # total_items = data_utils.get_total_item_size(info, "test")
        # labels = data_utils.get_labels(info)
        labels = ["bg"] + labels
        hyper_params["total_labels"] = len(labels)
        img_size = hyper_params["img_size"]

        data_shapes = data_utils.get_data_shapes()
        padding_values = data_utils.get_padding_values()

        test_data = test_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size, evaluate=True))

        test_data = test_data.padded_batch(self.batch_size, padded_shapes=data_shapes, padding_values=padding_values)

        prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])

        ssd_model = get_model(hyper_params)

        def evaluator(model_path):
            ssd_model.load_weights(model_path)

            ssd_decoder_model = get_decoder_model(ssd_model, prior_boxes, hyper_params)

            step_size = train_utils.get_step_size(total_items, self.batch_size)
            pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(test_data, steps=step_size, verbose=1)
            eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, self.batch_size)

        self.evaluator = evaluator

    def evaluate(self, model_path):
        self.evaluator(model_path)

    # def draw(self):
    #     drawing_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)

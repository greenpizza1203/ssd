import os
import argparse
import tensorflow as tf
from datetime import datetime


def get_model_path(model_type):
    """Generating model path from model_type value for save/load model weights.
    inputs:
        model_type = "vgg16", "mobilenet_v2"

    outputs:
        model_path = os model path, for example: "trained/ssd_vgg16_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "ssd_{}_model_weights.h5".format(model_type))
    return model_path

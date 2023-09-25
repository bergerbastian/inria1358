import os
import numpy as np

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = "/Users/paulrenger/code/Paukhard/inria1358/raw_data/aerial_images_inria1358" #os.path.join("..", "raw_data", "aerial_images_inria1358")
LOCAL_REGISTRY_PATH = "/Users/paulrenger/code/Paukhard/inria1358/mlops/training_outputs" #os.path.join("..", "mlops", "training_outputs")

LOCAL_API_DATA_FOLDER = "/Users/paulrenger/code/Paukhard/inria1358/raw_data/api"

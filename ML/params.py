import os
import numpy as np

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join("..", "raw_data", "aerial_images_inria1358")
LOCAL_REGISTRY_PATH =  os.path.join("..", "mlops", "training_outputs")

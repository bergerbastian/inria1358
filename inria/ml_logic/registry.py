import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras

import mlflow

from inria.params import *

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    if MODEL_TARGET == "mlflow":
        pass  # TBD

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        mlflow.log_params(params)
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        mlflow.log_metrics(metrics)
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None, custom_suffix = "") -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}{custom_suffix}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":

        # model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        # client = storage.Client()
        # bucket = client.bucket(BUCKET_NAME)
        # blob = bucket.blob(f"models/{model_filename}")
        # blob.upload_from_filename(model_path)

        # print("✅ Model saved to GCS")

        # return None
        pass

    if MODEL_TARGET == "mlflow":
        # mlflow.tensorflow.log_model(model=model,
        #                 artifact_path="model",
        #                 registered_model_name=MLFLOW_MODEL_NAME
        #                 )
        # print("✅ Model saved to mlflow")
        pass

    return None

def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk... ({most_recent_model_path_on_disk})" + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    # elif MODEL_TARGET == "gcs":
    #     print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

    #     client = storage.Client()
    #     blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    #     try:
    #         latest_blob = max(blobs, key=lambda x: x.updated)
    #         latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
    #         latest_blob.download_to_filename(latest_model_path_to_save)

    #         latest_model = keras.models.load_model(latest_model_path_to_save)

    #         print("✅ Latest model downloaded from cloud storage")

    #         return latest_model
    #     except:
    #         print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

    #         return None

    # elif MODEL_TARGET == "mlflow":
    #     print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

    #     # Load model from MLflow
    #     model = None
    #     pass  # YOUR CODE HERE
    #     return model
    else:
        return None

import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = None
db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID,
)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = None
model = ResNet50(include_top=True, weights="imagenet")

# Print the model summary
#model.summary()

def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # TODO

    # Loading the image with a target size of (224, 224)
    img_path = os.path.join(settings.UPLOAD_FOLDER, image_name)
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to a numpy array
    x = image.img_to_array(img)
    
    # Add a dimension to the array (as required by the model)
    x_batch = np.expand_dims(x, axis=0)
    # Preprocess the input image
    x_batch = preprocess_input(x_batch)

    preds = model.predict(x_batch)
    fst_pred = decode_predictions(preds, top=1)
    class_name = fst_pred[0][0][1]
    pred_probability = np.round(fst_pred[0][0][2], 4)

    return class_name, float(pred_probability)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO


        job = json.loads(db.brpop(settings.REDIS_QUEUE)[1].decode("utf-8"))

        job_id, image_name = job["id"], job["image_name"]
        class_name, pred_probability = predict(image_name)

        results = {"prediction": class_name, "score": pred_probability}

        db.set(job_id, json.dumps(results))

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)    


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()

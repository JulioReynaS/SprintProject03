import json
import os
import time

import numpy as np
import redis
import settings
<<<<<<< HEAD
=======
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
>>>>>>> 2ca48ea (Final commit)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
<<<<<<< HEAD
db = None
=======
>>>>>>> 2ca48ea (Final commit)
db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID,
<<<<<<< HEAD
=======
    decode_responses=True,
>>>>>>> 2ca48ea (Final commit)
)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
<<<<<<< HEAD
model = None
model = ResNet50(include_top=True, weights="imagenet")

# Print the model summary
#model.summary()

=======
model = ResNet50(include_top=True, weights="imagenet")

>>>>>>> 2ca48ea (Final commit)
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
<<<<<<< HEAD

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
=======
    try:
        # Load image
        img = image.load_img(
            os.path.join(settings.UPLOAD_FOLDER, image_name),
            target_size=(224, 224)
        )
        # Preprocess image
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Make predictions
        preds = model.predict(img)
        # Decode predictions
        class_name, pred_probability = decode_predictions(preds, top=1)[0][0][1:]

        pred_probability = float(round(pred_probability, 4))

    except:
        # If something goes wrong, print image name and return '0' for class_name
        class_name = image_name
        pred_probability = 0
        
    return class_name, pred_probability
>>>>>>> 2ca48ea (Final commit)


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

<<<<<<< HEAD

        job = json.loads(db.brpop(settings.REDIS_QUEUE)[1].decode("utf-8"))

        job_id, image_name = job["id"], job["image_name"]
        class_name, pred_probability = predict(image_name)

        results = {"prediction": class_name, "score": pred_probability}

        db.set(job_id, json.dumps(results))

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)    
=======
        # Get a job from the queue
        _, msg = db.brpop(settings.REDIS_QUEUE)

        if msg is not None:
            msg = json.loads(msg)
            # Run ML model
            class_name, pred_proba = predict(msg["image_name"])
            # Store predictions
            pred_proba = str(pred_proba)
            job_data = json.dumps({
                "prediction": class_name, 
                "score": pred_proba
            })
            # Store results on Redis
            db.set(msg["id"], job_data)
        
        else:
            continue

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)
>>>>>>> 2ca48ea (Final commit)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()

import io
import os
import flask
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array


app = flask.Flask(__name__)
# Placeholder for model
model = None


def model_load():
    """
    Load pretained model
    """
    global model
    path = 'model.h5'
    model = load_model(path)
    model._make_predict_function()


def prepare_image(image, target):
    """
    Preprocess image to use for prediction

    image: PIL image for input
    shape: Shape to resize image

    Returns:
    image: Numpy array of processed image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.
    return image


def make_prediction(image):
    """
    Make prediction of the image orientation

    image: PIL image for input

    Returns:
    scores: Probabilities of all possible orientation classes (0, 90, 180 ,270)
    orientation: Predicted orientation of the image
    """
    predictions = model.predict(image)[0]
    classes = (0, 90, 180, 270)
    scores = list(zip(classes, predictions))
    orientation = classes[np.argmax(predictions)]
    return scores, orientation


def rotate_image(image, rotation):
    """
    Rotate image to be right way up

    image: PIL image for input
    rotation: Degree to roate image

    Retrns:
    image: PIL image with correction rotation
    """
    return image.rotate(- rotation)


@app.route("/predict", methods=['POST'])
def predict():
    """
    Predict image orientation
    """
    if not model:
        model_load()
    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            molded_image = prepare_image(image, target=(224, 224))
            scores, orientation = make_prediction(molded_image)
            data["prediction"] = {}
            data['prediction']['orientation'] = orientation
            data['prediction']['scores'] = {}
            for rot, score in scores:
                data['prediction']['scores'][rot] = float(score)
            data["success"] = True
    return flask.jsonify(data)


@app.route("/correct", methods=['POST'])
def correct():
    """
    Correct image orientation
    """
    if not model:
        model_load()
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            molded_image = prepare_image(image, target=(224, 224))
            _, orientation = make_prediction(molded_image)
            image = rotate_image(image, orientation)
            mimetype = flask.request.files["image"].content_type
            extension = os.path.splitext(flask.request.files["image"].filename)[-1]
            file_name = 'rotated_image{}'.format(extension)
            image.save(file_name)
    return flask.send_file(file_name, mimetype=mimetype, as_attachment=True)


if __name__ == '__main__':
    model_load()
    app.run(host='0.0.0.0')

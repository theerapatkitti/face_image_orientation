# Face Image Orientation Classification and Correction

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* [Python 3.6](https://www.python.org/downloads/)
* [Docker](https://docs.docker.com/install/) 

### Setup

Install python modules

```
pip install -r requirements.txt
```

## Training

A pre-trained model is provided from the [releases page](https://github.com/theerapatkitti/face_image_orientation/releases). Training and evaluation code is in [face_orientation.py](face_orientation.py). The module can be ran directly from command line:

```
# Train a new model
python3 face_orientation.py train --dataset=/path/to/dataset/ --logs=logs

# Continue training a model that you had trained earlier
python3 face_orientation.py train --dataset=/path/to/dataset/ --logs=logs --model=/path/to/model.h5

# Continue training the last model you trained
python3 face_orientation.py train --dataset=/path/to/dataset/ --logs=logs --model=last
```

The training learning rate and other parameters should be set in face_orientation.

## Evaluation

Evaluation code can be ran by using:

```
# Run evaluation on the specified model
python3 face_orientation.py evaluate --dataset=/path/to/dataset/ --logs=logs --model=/path/to/model.h5
```

## Dataset

[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) is used as a dataset to train the model.

## API

There are two APIs available: orientation classification and orientation correction. Both APIs support JPEG, PNG, BMP.

### Orientation Classification

Orientation classification will classify image into one of the four classes of orientation: 0, 90, 180, and 270. The API will return the predicted orientation as well as probabilities of all four classes. The response will have a form:

```
{
    "prediction": {
        "orientation": ...,
        "scores": {
            "0": ...,
            "90": ...,
            "180": ...,
            "270": ...
        }
    }
    "success": ...
}
```

To call API, curl command can be used. 
NOTE: curl command does not show content-type of bitmap image as image/bmp so need to be done explicitly

```
# Run on local machine
curl -X POST -F image=@path/to/image "http://localhost:5000/predict"

# Run on cloud
curl -X POST -F image=@path/to/image "http://<external ip>/predict"
```

Or by using python script.

```
import requests
files = {"image": ("filename", open("path/to/image", "rb"), "mimetype")}

# Run on local machine
requests.post("http://localhost:5000/predict", files=files)

# Run on cloud
requests.post("http://<external ip>/predict", files=files)
```

### Orientation Correction

Orientation correction will correct image orientation to make the image upright. The API will return the upright image.

To call API, curl command can be used. 
NOTE: curl command does not show content-type of bitmap image as image/bmp so need to be done explicitly

```
# Run on local machine
curl -X POST -F image=@path/to/image "http://localhost:5000/correct"

# Run on cloud
curl -X POST -F image=@path/to/image "http://<external ip>/correct"
```

Or by using python script.

```
import requests
files = {"image": ("filename", open("path/to/image", "rb"), "mimetype")}

# Run on local machine
requests.post("http://localhost:5000/correct", files=files)

# Run on cloud
requests.post("http://<external ip>/correct", files=files)
```

## Testing

There are two tests within the project: testing API on local machine, and testing API of deployed model.

### Test API on Local Machine

This test is for checking API on local machine before the model is deployed. The test will check the response of API.

Before calling the script, the Flask app must be ran first.

```
cd docker

python app.py
```

After the Flask app is running, the test script can be used.

```
cd test

python test_local.py
```

### Test API of Deployed model

The test is for checking API of deployed model to make sure it is processing correctly.

```
cd test

python test_api.py
```

## Deployment

Docker, Flask, and Google Cloud Platform is used to develop and deploy the API. Flask is used to create an API while Docker and Kubernetes is for scalable deployment.

To deploy the API, a docker image is created.

```
cd docker

sudo docker build -t face-orientation-api:latest .
```

Then the docker container is uploaded to Docker Hub for Kubernetes to install.

```
sudo docker login

sudo docker tag <image id> <docker hub id>/<app name>

sudo docker push <docker hub id>/<app-name>
```

Create Kubernetes cluster using Google Kubernetes Engine and run the docker container in Kubernetes.

```
kubectl run <app name> --image=<docker hub id>/<app name> --port 5000

kubectl expose deployment <app name> --type=LoadBalancer --port 80 --target-port 5000
```

## Built With

* [Keras](https://keras.io/)
* [Docker](https://www.docker.com/)
* [Flask](http://flask.pocoo.org/)
* [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/docs/)

## Citation

[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)

```
@TechReport{LFWTech,
  author =       {Gary B. Huang and Manu Ramesh and Tamara Berg and 
                  Erik Learned-Miller},
  title =        {Labeled Faces in the Wild: A Database for Studying 
                  Face Recognition in Unconstrained Environments},
  institution =  {University of Massachusetts, Amherst},
  year =         2007,
  number =       {07-49},
  month =        {October}}
```
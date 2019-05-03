import os
import sys
import time
import imgaug
import skimage
import numpy as np
from utils.dataset import Dataset
from utils.model import Model


class Config():
    NUM_CLASSES = 4
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 32
    MAX_EPOCH = 10
    LEARNING_RATE = 0.001


def evaluate(model, dataset, limit=0):
    """Runs evaluation.

    model: model to evaluate
    dataset: Dataset object with valiadtion data
    limit: number of images to use for evaluation
    """
    # Pick images from the dataset
    image_ids = dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    t_prediction = 0
    t_start = time.time()
    acc = 0

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        image = skimage.transform.resize(image, config.INPUT_SHAPE, 
                                        mode="constant", anti_aliasing=True)
        class_id = dataset.image_info[image_id]["class"]
        image = skimage.transform.rotate(image, dataset.class_info[class_id]["name"])

        # Run prediction
        t = time.time()
        r = model.predict(np.expand_dims(image, axis=0))[0]
        t_prediction += (time.time() - t)

        acc += np.argmax(r) == class_id

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    print("Accuracy: ", acc / len(image_ids))


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train face orientation")
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = Config()
    model = Model(config=config, model_dir=args.logs)

    if args.model:
        # Select weights file to load
        if args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()
        else:
            model_path = args.model
    
        model.load_model(model_path)

    elif args.command == "train":
        model.build(config)
    
    model.model.summary()

     # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = Dataset()
        dataset_train.load_dataset(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = Dataset()
        dataset_val.load_dataset(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training
        print("Training network")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=config.MAX_EPOCH,
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = Dataset()
        dataset_val.load_dataset(args.dataset, "val")
        dataset_val.prepare()
        print("Running evaluation on {} images.".format(args.limit))
        evaluate(model, dataset_val, limit=int(args.limit))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
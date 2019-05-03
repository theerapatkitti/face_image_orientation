import os
import sys
import imgaug
from utils.dataset import Dataset
from utils.model import Model


class Config():
    NUM_CLASSES = 4
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 32
    MAX_EPOCH = 10
    LEARNING_RATE = 0.001
    MODEL = "ResNet50"

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

    if args.command == "train":
        model.build(config)

    if args.model:
        # Select weights file to load
        if args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()
        else:
            model_path = args.model
    
        model.load_model(model_path)
    
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
        print("Running evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
import os
import json
import random


def make_annotation(paths):
    annotations = []
    for i, path in enumerate(paths):
        annotations.append({
            "id": "%06d" % i,
            "path": path
        })
    return annotations


def prepare_dataset(dataset_dir):
    paths = []
    for folder in os.listdir(dataset_dir):
        folder_dir = os.path.join(dataset_dir, folder)
        for filename in os.listdir(folder_dir):
            paths.append(os.path.join(folder_dir, filename))
    
    n = int(len(paths) * 0.9)
    random.shuffle(paths)
    train = make_annotation(paths[:n])
    val = make_annotation(paths[n:])
    
    annotation_dir = os.path.join(dataset_dir, "annotations")
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)

    with open(os.path.join(annotation_dir, "train.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(annotation_dir, "val.json"), "w") as f:
        json.dump(val, f)

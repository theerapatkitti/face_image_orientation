import os
import json
import skimage
import numpy as np
from dataset import lfw_dataset


class Dataset():
    def __init__(self):
        self.image_ids = []
        self.image_info = []
        self.class_info = []
    
    def load_dataset(self, dataset_dir, subset):
        self.add_class(0, 0)
        self.add_class(1, 90)
        self.add_class(2, 180)
        self.add_class(3, 270)
        
        annotation_dir = os.path.join(dataset_dir, "annotations/{}.json".format(subset))
        if not os.path.exists(annotation_dir):
            lfw_dataset.prepare_dataset(dataset_dir)

        with open(annotation_dir, "r") as f:
            annotations = json.load(f)
        
        for a in annotations:
            class_id = np.random.randint(4)
            self.add_image(a["id"], a["path"], class_id)

    def add_class(self, class_id, class_name):
        for info in self.class_info:
            if info["id"] == class_id:
                # class_id already available, skip
                return
        # Add the class
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path, class_id):
        image_info = {
            "id": image_id,
            "path": path,
            "class": class_id
        }
        self.image_info.append(image_info)
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def prepare(self):
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
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
        """Load the dataset

        dataset_dir: path to the dataset
        subset: train or val
        """
        for i in range(4):
            self.add_class(i, 360 // 4 * i)
        
        annotation_dir = os.path.join(dataset_dir, "annotations/{}.json".format(subset))
        if not os.path.exists(annotation_dir):
            lfw_dataset.prepare_dataset(dataset_dir)

        with open(annotation_dir, "r") as f:
            annotations = json.load(f)
        
        for a in annotations:
            class_id = np.random.randint(4)
            self.add_image(a["id"], a["path"], class_id)

    def add_class(self, class_id, class_name):
        """ Add class id.

        class_id: integer of class id
        class_name: name of the class
        """
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
        """ Add image to the Dataset object.

        image_id: ID of the image
        path: path to the image file
        class_id: class id of the image
        """
        image_info = {
            "id": image_id,
            "path": path,
            "class": class_id
        }
        self.image_info.append(image_info)
    
    def load_image(self, image_id):
        """Load the image and return a [H,W,3] Numpy array.

        image_id: ID of the image to load
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
        """Prepares the Dataset class.
        """
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
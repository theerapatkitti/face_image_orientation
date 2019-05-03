import os
import re
import datetime
import logging
import threading
import skimage
import numpy as np
import keras
import keras.layers as KL
import keras.models as KM
import keras.optimizers as KO
from keras.applications.resnet50 import ResNet50


def load_image(dataset, config, image_id, augmentation=None):
    """Load image and class.

    dataset: The Dataset object to pick data from
    config: The model config object
    image_id: ID of the image to load
    augmentation: An imgaug (https://github.com/aleju/imgaug) augmentation (Optional)
    
    Returns:
    image: [height, width, 3]
    class_ids: Integer class ID
    """
    # Load image
    image = dataset.load_image(image_id)
    image = skimage.transform.resize(image, config.INPUT_SHAPE, 
                                        mode="constant", anti_aliasing=True)
    class_id = dataset.image_info[image_id]["class"]

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        # Store shapes before augmentation to compare
        image_shape = image.shape
        # Make augmenters deterministic to apply similarly to images
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"

    image = skimage.transform.rotate(image, dataset.class_info[class_id]["name"])

    return image, class_id


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def data_generator(dataset, config, shuffle=True, augmentation=None, batch_size=1):
    """A generator that returns images and corresponding target class ids.

    dataset: The Dataset object to pick data from
    config: The model config object
    augmentation: An imgaug (https://github.com/aleju/imgaug) augmentation (Optional)
    batch_size: Number of images to use in each call

    Returns:
    inputs: lists containing images
    - images : [batch, H, W, C]
    outputs: lists containing corresponding target class ids
    - class_ids : Integer class IDs
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]

            image, class_id = load_image(dataset, config, image_id,
                                augmentation=augmentation)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_class_ids = np.zeros(
                    (batch_size, config.NUM_CLASSES), dtype=np.int32)

            # Add to batch
            batch_images[b] = image.astype(np.float32)
            batch_class_ids[b, class_id] = 1
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images]
                outputs = [batch_class_ids]

                yield inputs, outputs

                # start a new batch
                b = 0
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


class Model():
    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
    
    def build(self, config):
        """Build ResNet50 architecture.
        
        config: The model config object
        """
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=config.INPUT_SHAPE)
        x = base_model.output
        x = KL.Flatten()(x)
        final_output = KL.Dense(config.NUM_CLASSES, activation='softmax', name='rotation')(x)

        model = KM.Model(inputs=base_model.input, outputs=final_output)
        
        self.model = model
    
    def train(self, train_dataset, val_dataset, learning_rate, epochs,
                augmentation=None, custom_callbacks=None):
        """Train the model.

        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs.
        augmentation: An imgaug (https://github.com/aleju/imgaug) augmentation (Optional)
        custom_callbacks: list of type keras.callbacks to be called
	        with the keras fit_generator method (Optional)
        """
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        self.model.compile(
            optimizer=KO.Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['acc']
        )

        self.model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            verbose=1,
            steps_per_epoch=train_dataset.num_images / self.config.BATCH_SIZE,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=val_dataset.num_images / self.config.BATCH_SIZE,
            workers=10,
        )
        self.epoch = max(self.epoch, epochs)

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.

        Returns:
        checkpoint: The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("model"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_model(self, filepath):
        """Load pre-trained model from file

        filepath: path to pre-trained model file
        """
        self.model = KM.load_model(filepath)
        
        # Update the log directory
        self.set_log_dir(filepath)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: path to last checkpoint (Optional)
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\model20171029T2315\model_0001.h5 (Windows)
            # /path/to/logs/model20171029T2315/model_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]model\_(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "model{:%Y%m%dT%H%M}".format(now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "model_*epoch*.h5")
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
            
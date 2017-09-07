import scipy.misc
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
from keras import backend as K
from keras.preprocessing import image

class MyImageDataGenerator(ImageDataGenerator):

    def flow(self, file, image_size, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return FileIterator(
            file, image_size, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class FileIterator(Iterator):
    """Iterator yielding data from a file.

    The file should be in the following format:
    <image_path_1> <label_data_1>
    <image_path_2> <label_data_2>
    ...
    <image_path_n> <label_data_n>

    # Arguments
        file: Path to the file to read the image list and label data.
        image_size: Image size, [height, width, channel]
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, file, image_size, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if not os.path.exists(file):
            raise ValueError('Cannot find file: %s' % file)

        if data_format is None:
            data_format = K.image_data_format()

        split_lines = [line.rstrip('\n').split(' ') for line in open(file, 'r')]
        self.x = np.asarray([e[0] for e in split_lines])
        self.y = np.asarray([float(e[1]) for e in split_lines])
        self.image_size = image_size
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(FileIterator, self).__init__(self.x.shape[0], batch_size, shuffle, seed)

    def next(self):
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.image_size)), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = scipy.misc.imread(self.x[j])
            x = scipy.misc.imresize(x, self.image_size)
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = image.array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        batch_y = self.y[index_array]
        return batch_x, batch_y
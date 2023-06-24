import os
import imageio
import numpy as np

from scipy import misc
from skimage import color


class ValidationSet:
    def __init__(self, image_dir=os.path.join(os.path.dirname(__file__), 'data', 'ILSVRC2012_img_val'),
                 labels_path=os.path.join(os.path.dirname(__file__), 'data', 'ILSVRC2012_validation_ground_truth.txt'),
                 scaling_factor=1, size=(224, 224)):
        print(image_dir,labels_path)
        assert os.path.exists(image_dir)

        assert os.path.exists(labels_path)

        self.image_dir = image_dir
        self.labels_path = labels_path
        self.scaling_factor = scaling_factor
        self.size = size
        self.current_image = 0
        self.image_paths = []
        self.labels = []

        with open(self.labels_path) as f:
            for line in f:
                file_name, label = line.split(' ')
                image_path = os.path.join(self.image_dir, file_name)

                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_image < len(self.image_paths):
            self.current_image += 1

            image = imageio.imread(self.image_paths[self.current_image - 1])

            if self.scaling_factor > 1:
                image = lower_resolution(image, self.scaling_factor)

            if len(image.shape) == 2:
                image = np.stack((image, ) * 3, axis=-1)

            if self.size is not None:
                x = int(image.shape[0] * self.size[0] / np.min(image.shape[0:2]))
                y = int(image.shape[1] * self.size[1] / np.min(image.shape[0:2]))
                image = misc.imresize(image, (x, y), 'bicubic')
                x_start = int((x - self.size[0]) / 2)
                x_end = int((x - self.size[0]) / 2 + self.size[0])
                y_start = int((y - self.size[1]) / 2)
                y_end = int((y - self.size[1]) / 2 + self.size[1])
                image = image[x_start:x_end, y_start:y_end]

            return image
        else:
            self.current_image = 0

            raise StopIteration


def lower_resolution(image, scaling_factor):
    assert image.dtype == 'uint8'

    width = image.shape[0]
    height = image.shape[1]
    width = width - width % scaling_factor
    height = height - height % scaling_factor
    image = image[:width, :height]

    if len(image.shape) == 3:
        image_ycbcr = color.rgb2ycbcr(image)
        image_y = image_ycbcr[:, :, 0].astype(np.uint8)
    else:
        image_y = image.copy()

    downscaled = misc.imresize(image_y, 1 / float(scaling_factor), 'bicubic', mode='L')
    rescaled = misc.imresize(downscaled, float(scaling_factor), 'bicubic', mode='L').astype(np.float32)

    if len(image.shape) == 3:
        low_res_image = image_ycbcr
        low_res_image[:, :, 0] = rescaled
        low_res_image = color.ycbcr2rgb(low_res_image)
        low_res_image = (np.clip(low_res_image, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        low_res_image = rescaled.astype(np.uint8)

    return low_res_image

import os
import imageio

from containers import ValidationSet


for scaling_factor in [2, 3, 4]:
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'ILSVRC2012_img_val_LR_scale_factor_%d' % scaling_factor)

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    dataset = ValidationSet(scaling_factor=scaling_factor, size=None)

    for image_path, image in zip(dataset.image_paths, dataset):
        file_name = os.path.basename(image_path).replace('.JPEG', '.TIF')
        imageio.imwrite(os.path.join(dataset_path, file_name), image)

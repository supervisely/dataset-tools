import os

import numpy as np


class Previews:
    def __init__(self, project_data):
        self.project_meta = project_data.project_meta
        self.datasets = project_data.datasets
        self.max_width = 500

    def save(self, save_dir):
        for dataset in self.datasets:
            image_names = [image_info.name for image_info in dataset.image_infos]

            for image_name, ann in zip(image_names, dataset.anns):
                self._resize_and_save_image(save_dir, dataset.name, image_name, ann)

    def _resize_and_save_image(self, save_dir, dataset_name, image_name, ann):
        height, width = ann.img_size

        if width > self.max_width:
            out_size = (self.max_width, int((width / height) * self.max_width))
            resized_ann = ann.resize(out_size)
        else:
            out_size = (width, height)
            resized_ann = ann

        image_filename = os.path.splitext(image_name)[0] + ".png"

        image_save_path = os.path.join(save_dir, dataset_name, image_filename)

        image = np.zeros((out_size[0], out_size[1], 3), dtype=np.uint8)

        resized_ann.draw_pretty(image, output_path=image_save_path, opacity=1)

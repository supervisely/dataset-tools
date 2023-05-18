import os
import random
from typing import Union
import cv2
from supervisely.imaging import font as sly_font

import numpy as np
from tqdm import tqdm

import supervisely as sly


class VerticalGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 6,
        cols: int = 3,
    ):
        self.project_meta = project_meta

        self._max_width = 1920
        self._rows = rows
        self._cols = cols
        self._g_spacing = 15

        self._all_image_infos = []
        self._all_anns = []
        self.np_images = []
        self._grid = None
        self._column_width = (self._max_width - self._g_spacing * (self._cols + 1)) / self._cols

        self._local = False if isinstance(project, int) else True
        self._api = api if api is not None else sly.Api.from_env()

    def update(self, data: tuple):
        cnt = self._cols * self._rows
        join_data = [(ds, img, ann) for ds, list1, list2 in data for img, ann in zip(list1, list2)]

        random.shuffle(join_data)
        with tqdm(desc="Downloading images", total=cnt) as p:
            for ds, img_info, ann in join_data[:cnt]:
                ann: sly.Annotation
                self._all_image_infos.append(img_info)
                self._all_anns.append(ann)
                img = (
                    sly.image.read(ds.get_img_path(img_info.name))
                    if self._local
                    else self._api.image.download_np(img_info.id)
                )
                p.update(1)

                ann.draw_pretty(img, thickness=0, opacity=0.3)
                img = self._resize_image(img)
                self.np_images.append(img)

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            sly.fs.clean_dir(storage_dir)
            path = os.path.join(storage_dir, "separated_images_grid.jpeg")
        sly.image.write(path, self._grid)
        sly.logger.info(f"Result grid saved to: {path}")

    def _create_image_canvas(self, columns):
        heights = []
        for column in columns:
            clmn_h = column.shape[:2][0]
            heights.append(clmn_h)
        grid_h = max(heights)
        self._grid = np.ones([grid_h, self._max_width, 3], dtype=np.uint8) * 255

    def _merge_canvas_with_images(self):
        columns = self._create_columns()
        self._create_image_canvas(columns)
        for i, image in enumerate(columns):
            self._grid[:, i * self._column_width : (i + 1) * self._column_width] = np.expand_dims(
                image, axis=1
            )

    def _create_columns(self):
        num_images = len(self.np_images)
        image_heights = [image.shape[0] for image in self.np_images]

        one_big_column_height = sum(image_heights) + (num_images - 1) * self._g_spacing
        target_column_height = one_big_column_height // self._cols

        columns = []
        column_images = []
        current_height = 0

        for image, height in zip(self.np_images, image_heights):
            if current_height + height > target_column_height:
                column_images.append(image)
                columns.append(column_images)

                column_images = []
                current_height = 0

            column_images.append(image)
            current_height += height + self._g_spacing

        columns.append(column_images)

        return columns

    def _resize_image(self, image):
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = self._column_width / img_w
        img_w = self._column_width
        img_h = img_aspect_ratio * img_h

        image = sly.image.resize(image, (img_h, img_w))

        return image

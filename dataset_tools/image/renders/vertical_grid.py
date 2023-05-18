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
        footer_path: str = "dninja_footer.png",
    ):
        self.project_meta = project_meta

        self._img_width = 1920
        self._rows = rows
        self._cols = cols
        self._g_spacing = 15
        self._column_height = 0
        self._footer_path = footer_path

        self._all_image_infos = []
        self._all_anns = []
        self.np_images = []
        self._img_array = None
        self._column_width = int(
            (self._img_width - self._g_spacing * (self._cols + 1)) / self._cols
        )

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

                ann.draw_pretty(img, thickness=0, opacity=0.7)
                img = self._resize_image(img)
                self.np_images.append(img)

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            path = os.path.join(storage_dir, "vertical_grid.png")
        self._merge_canvas_with_images()
        self._add_footer_with_logo()
        cv2.imwrite(path, self._img_array)
        sly.logger.info(f"Result grid saved to: {path}")

    def _create_image_canvas(self):
        self._img_array = np.ones([self._column_height, self._img_width, 3], dtype=np.uint8) * 255

    def _merge_canvas_with_images(self):
        columns = self._create_columns()
        self._create_image_canvas()
        columns = self._merge_img_in_columns(columns)
        for i, image in enumerate(columns):
            if image.shape[0] > self._img_array.shape[0]:
                image = image[: self._img_array.shape[0] - self._g_spacing, :]

                column_start = i * (self._column_width + self._g_spacing) + self._g_spacing
                column_end = column_start + self._column_width
                row_start = self._g_spacing
                row_end = self._img_array.shape[0]

                self._img_array[row_start:row_end, column_start:column_end] = image

    def _create_columns(self):
        num_images = len(self.np_images)
        image_heights = [image.shape[0] for image in self.np_images]

        one_big_column_height = sum(image_heights) + (num_images - 1) * self._g_spacing
        self._column_height = one_big_column_height // self._cols

        columns = []
        column_images = []
        current_height = 0

        for image, height in zip(self.np_images, image_heights):
            if current_height + height > self._column_height:
                column_images.append(image)
                columns.append(column_images)

                column_images = []
                current_height = 0

            column_images.append(image)
            current_height += height + self._g_spacing

        if len(columns) == self._cols:
            return columns
        return columns

    def _merge_img_in_columns(self, columns):
        combined_columns = []
        separator = np.ones((15, self._column_width, 3), dtype=np.uint8) * 255
        for column in columns:
            combined_images = []

            for image in column:
                combined_images.append(image)
                combined_images.append(separator)
            combined_images.pop()
            combined_image = np.vstack(combined_images)
            combined_columns.append(combined_image)

        return combined_columns

    def _resize_image(self, image):
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = self._column_width / img_w
        img_w = int(self._column_width)
        img_h = int(img_aspect_ratio * img_h)

        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _add_footer_with_logo(self):
        image2 = cv2.imread(self._footer_path, cv2.IMREAD_UNCHANGED)

        height1, _ = self._img_array.shape[:2]
        height2, width2 = image2.shape[:2]

        alpha_channel = image2[:, :, 3] / 255.0

        x = 0
        y = height1 - height2

        region = self._img_array[y:height1, x : x + width2]
        self._img_array[y:height1, x : x + width2, :3] = (
            1 - alpha_channel[:, :, np.newaxis]
        ) * region + alpha_channel[:, :, np.newaxis] * image2[:, :, :3]

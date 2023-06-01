import os
import random
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

import supervisely as sly
from dataset_tools.image.renders.convert import compress_png


class SideAnnotationsGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 3,
        cols: int = 3,
        side_overlay_path: str = "side_logo_overlay.png",
        force: bool = False,
    ):
        self.force = force
        self.project_meta = project_meta

        self._img_height = 1080
        self._rows = rows
        self._cols = cols
        self._row_width = 0
        self._gap = 15
        self._bg_color = (221, 210, 230)  # rgb(221, 210, 230)
        self._side_overlay_path = side_overlay_path

        self._local = False if isinstance(project, int) else True
        self._api = api if api is not None else sly.Api.from_env()

        self.np_images = []
        self._img_array = None
        self._row_height = int((self._img_height - self._gap * (self._rows + 1)) / self._rows)

    @property
    def basename_stem(self) -> None:
        return sly.utils.camel_to_snake(self.__class__.__name__)

    def update(self, data: tuple):
        cnt = self._cols * self._rows
        join_data = [(ds, img, ann) for ds, list1, list2 in data for img, ann in zip(list1, list2)]

        random.shuffle(join_data)
        with tqdm(desc="Downloading images", total=cnt) as p:
            for ds, img_info, ann in join_data[:cnt]:
                img = (
                    sly.image.read(ds.get_img_path(img_info.name))
                    if self._local
                    else self._api.image.download_np(img_info.id)
                )
                mask = self._draw_masks_on_single_image(ann, img_info)

                img = self._resize_image(img)
                mask = self._resize_image(mask)
                join_image = np.hstack([img, mask])
                self.np_images.append(join_image)

                p.update(1)

    def _draw_masks_on_single_image(self, ann: sly.Annotation, image: sly.ImageInfo):
        height, width = image.height, image.width
        mask_img = np.zeros((height, width, 3), dtype=np.uint8)
        mask_img[:, :, 0:3] = self._bg_color  # rgb(221, 210, 230)

        for label in ann.labels:
            label.geometry.draw(mask_img, color=label.obj_class.color, thickness=3)

        return mask_img

    def to_image(self, path: str = None):
        path_part, ext = os.path.splitext(path)
        tmp_path = f"{path_part}-o{ext}"
        if path is None:
            storage_dir = sly.app.get_data_dir()
            path = os.path.join(storage_dir, "horizontal_grid.png")

        self._merge_canvas_with_images()
        self._add_overlay_with_logo()

        sly.image.write(tmp_path, self._img_array)
        compress_png(tmp_path, path)
        sly.fs.silent_remove(tmp_path)
        sly.logger.info(f"Result grid saved to: {path}")

    def _create_image_canvas(self):
        self._img_array = np.ones([self._img_height, self._row_width, 3], dtype=np.uint8) * 255

    def _merge_canvas_with_images(self):
        rows = self._create_rows()
        self._create_image_canvas()
        rows = self._merge_img_in_rows(rows)
        for i, image in enumerate(rows):
            if image.shape[1] > self._img_array.shape[1]:
                image = image[:, : self._img_array.shape[1] - self._gap]

            row_start = i * (self._row_height + self._gap) + self._gap
            row_end = row_start + self._row_height
            column_start = self._gap
            column_end = self._img_array.shape[1]

            self._img_array[row_start:row_end, column_start:column_end] = image

    def _create_rows(self):
        num_images = len(self.np_images)
        image_widths = [image.shape[1] for image in self.np_images]

        one_big_row_width = sum(image_widths) + (num_images - 1) * self._gap
        self._row_width = one_big_row_width // self._rows

        rows = []
        row_images = []
        current_width = 0

        for image, width in zip(self.np_images, image_widths):
            if current_width + width > self._row_width:
                row_images.append(image)
                rows.append(row_images)

                row_images = []
                current_width = 0

            row_images.append(image)
            current_width += width + self._gap

        if len(rows) == self._rows:
            return rows
        return rows

    def _merge_img_in_rows(self, rows):
        combined_rows = []
        separator = np.ones((self._row_height, 15, 3), dtype=np.uint8) * 255
        for row in rows:
            combined_images = []

            for image in row:
                combined_images.append(image)
                combined_images.append(separator)
            combined_images.pop()
            combined_image = np.hstack(combined_images)
            combined_rows.append(combined_image)

        return combined_rows

    def _resize_image(self, image):
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = self._row_height / img_h
        img_h = int(self._row_height)
        img_w = int(img_aspect_ratio * img_w)

        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _add_overlay_with_logo(self):
        image2 = cv2.imread(self._side_overlay_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGBA)

        _, width1 = self._img_array.shape[:2]
        height2, width2 = image2.shape[:2]

        alpha_channel = image2[:, :, 3] / 255.0

        x = width1 - width2

        region = self._img_array[:height2, x : x + width2]
        self._img_array[:height2, x : x + width2, :3] = (
            1 - alpha_channel[:, :, np.newaxis]
        ) * region + alpha_channel[:, :, np.newaxis] * image2[:, :, :3]

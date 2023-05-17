import os
import random
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

import supervisely as sly
from supervisely.imaging import font as sly_font


class HorizontalGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 3,
        cols: int = 3,
    ):
        self.project_meta = project_meta

        self._max_size = 1920
        self._rows = rows
        self._cols = cols
        self._aspect_ratio = 9 / 16
        self._gap = 15

        height, width, piece_h, piece_w = self._calculate_shapes()
        self._grid_size = (height, width)
        self._piece_size = (piece_h, piece_w)

        self._all_image_infos = []
        self._all_anns = []
        self.np_images = []
        self._grid = None

        self._local = False if isinstance(project, int) else True
        self._api = api if api is not None else sly.Api.from_env()

    @property
    def render_name(self) -> None:
        return sly.utils.camel_to_snake(self.__class__.__name__)

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
                img = self._resize_image(img)
                ann = ann.resize(self._piece_size)
                ann.draw_pretty(img, thickness=0, opacity=0.3)
                for label in ann.labels:
                    bbox = label.geometry.to_bbox()
                    cv2.rectangle(
                        img,
                        (bbox.left, bbox.top),
                        (bbox.right, bbox.bottom),
                        color=label.obj_class.color,
                        thickness=2,
                    )
                    font_size = sly_font.get_readable_font_size(img.shape[:2]) * 2
                    font = sly_font.get_font(font_size=font_size)
                    _, _, _, bottom = font.getbbox(label.obj_class.name)
                    anchor = (bbox.top - bottom, bbox.left)
                    sly.image.draw_text(img, label.obj_class.name, anchor, font=font)

                self.np_images.append(img)

        self._grid = self._create_image_grid(self.np_images)

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            sly.fs.clean_dir(storage_dir)
            path = os.path.join(storage_dir, "separated_images_grid.jpeg")
        sly.image.write(path, self._grid)
        sly.logger.info(f"Result grid saved to: {path}")

    def _calculate_shapes(self):
        height = width = self._max_size
        if self._rows > self._cols:
            piece_h = (height - self._gap * (self._rows + 1)) // self._rows
            piece_w = int(max(1, piece_h / self._aspect_ratio))
        else:
            piece_w = (width - self._gap * (self._cols + 1)) // (self._cols)
            piece_h = int(max(1, piece_w * self._aspect_ratio))
        height = (piece_h + self._gap) * self._rows + self._gap
        width = (piece_w + self._gap) * self._cols + self._gap

        sly.logger.info(f"Grid item size is (h: {piece_h}, w: {piece_w})")
        return height, width, piece_h, piece_w

    def _create_image_grid(self, images):
        img_h, img_w = images[0].shape[:2]
        num = len(images)

        grid = np.ones([*self._grid_size, 3], dtype=np.uint8) * 255

        for idx in range(num):
            x = (idx % self._cols) * (img_w + self._gap) + self._gap
            y = (idx // self._cols) * (img_h + self._gap) + self._gap
            grid[y : y + img_h, x : x + img_w, ...] = images[idx][:, :, ...]

        return grid

    def _resize_image(self, image):
        h, w = self._piece_size

        img_h, img_w = image.shape[:2]
        src_ratio = w / h
        img_ratio = img_w / img_h
        if img_ratio == src_ratio:
            image = sly.image.resize(image, (h, w))
        else:
            if img_ratio < src_ratio:
                img_h, img_w = int((w * img_h) // img_w), w
            else:
                img_h, img_w = h, int((h * img_w) // img_h)
            image = sly.image.resize(image, (img_h, img_w))
            crop_rect = sly.Rectangle(
                top=(img_h - h) // 2,
                left=(img_w - w) // 2,
                bottom=(img_h + h) // 2,
                right=(img_w + w) // 2,
            )
            image = sly.image.crop_with_padding(image, crop_rect)
            image = sly.image.resize(image, (h, w))

        return image

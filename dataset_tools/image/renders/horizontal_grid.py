import os
import random
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import supervisely as sly
from supervisely.imaging import font as sly_font


class HorizontalGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 3,
        cols: int = 6,
        side_overlay_path: str = "side_logo_overlay.png",
    ):
        self.project_meta = project_meta

        self._img_height = 1080
        self._rows = rows
        self._cols = cols
        self._gap = 15
        self._row_width = 0
        self._side_overlay_path = side_overlay_path

        self.np_images = []  # for grid
        self.np_anns = []  # for gif
        self.np_frames = []  # for gif
        self._img_array = None
        self._row_height = int((self._img_height - self._gap * (self._rows + 1)) / self._rows)

        self._local = False if isinstance(project, int) else True
        self._api = api if api is not None else sly.Api.from_env()

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
                p.update(1)
                ann: sly.Annotation
                tmp = np.dstack((img, np.ones((*img.shape[:2], 1), dtype=np.uint8) * 255))
                ann_mask = np.ones((*img.shape[:2], 4), dtype=np.uint8) * 255
                ann.draw_pretty(ann_mask[:, :, :3], thickness=0, opacity=0.7)
                self.np_frames.append(self._resize_image(tmp, self._row_height))  # for gif
                self.np_anns.append(self._resize_image(ann_mask, self._row_height))  # for gif

                ann.draw_pretty(img, thickness=0, opacity=0.7)  # for grid
                for label in ann.labels:
                    if label.obj_class.name == "neutral":
                        continue
                    bbox = label.geometry.to_bbox()
                    cv2.rectangle(
                        img,
                        (bbox.left, bbox.top),
                        (bbox.right, bbox.bottom),
                        color=label.obj_class.color,
                        thickness=2,
                    )
                    font_size = int(sly_font.get_readable_font_size(img.shape[:2]) * 1.4)
                    font = sly_font.get_font(font_size=font_size)
                    _, _, _, bottom = font.getbbox(label.obj_class.name)
                    anchor = (bbox.top - bottom, bbox.left)
                    sly.image.draw_text(img, label.obj_class.name, anchor, font=font)
                img = self._resize_image(img, self._row_height)
                self.np_images.append(img)

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            path = os.path.join(storage_dir, "horizontal_grid.png")

        self._img_array = self._merge_canvas_with_images(self.np_images)
        self._add_overlay_with_logo(self._img_array)
        sly.image.write(path, self._img_array)
        sly.logger.info(f"Result grid saved to: {path}")

    def _merge_canvas_with_images(self, images, channels: int = 3):
        rows = self._create_rows(images)
        rows = self._merge_img_in_rows(rows, channels)

        canvas = np.ones([self._img_height, self._row_width, channels], dtype=np.uint8) * 255
        for i, image in enumerate(rows):
            if image.shape[1] > canvas.shape[1] - self._gap:
                image = image[:, : canvas.shape[1] - self._gap]

            row_start = i * (self._row_height + self._gap) + self._gap
            row_end = row_start + self._row_height
            column_start = self._gap
            column_end = canvas.shape[1]

            canvas[row_start:row_end, column_start:column_end] = image
        return canvas

    def _create_rows(self, images):
        num_images = len(images)
        image_widths = [image.shape[1] for image in images]

        one_big_row_width = sum(image_widths[: -self._rows]) + (num_images - 1) * self._gap
        self._row_width = one_big_row_width // self._rows

        rows = []
        row_images = []
        current_width = 0

        for image, width in zip(images, image_widths):
            row_images.append(image)
            current_width += width + self._gap
            if current_width > self._row_width:
                rows.append(row_images)

                row_images = []
                current_width = 0

        if len(rows) == self._rows:
            return rows
        return rows

    def _merge_img_in_rows(self, rows, channels: int = 3):
        combined_rows = []
        separator = np.ones((self._row_height, 15, channels), dtype=np.uint8) * 255
        for row in rows:
            combined_images = []

            for image in row:
                combined_images.append(image)
                combined_images.append(separator)
            combined_images.pop()
            combined_image = np.hstack(combined_images)
            combined_rows.append(combined_image)

        return combined_rows

    def _resize_image(self, image, height):
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = height / img_h
        img_h = height
        img_w = int(img_aspect_ratio * img_w)

        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _add_overlay_with_logo(self, image):
        image2 = cv2.imread(self._side_overlay_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGBA)

        height1, width1 = image.shape[:2]
        height2, width2 = image2.shape[:2]
        if height1 != height2:
            scale_factor = height1 / height2
            height2, width2 = height1, int(scale_factor * width2)
            image2 = sly.image.resize(image, (height2, width2))
        alpha_channel = image2[:, :, 3] / 255.0

        x = width1 - width2

        region = image[:height2, x : x + width2]
        image[:height2, x : x + width2, :3] = (
            1 - alpha_channel[:, :, np.newaxis]
        ) * region + alpha_channel[:, :, np.newaxis] * image2[:, :, :3]

    def to_gif(self, path: str = None):
        bg = self._merge_canvas_with_images(self.np_frames, 4)
        bg = self._resize_image(bg, bg.shape[0] // 2)
        ann = self._merge_canvas_with_images(self.np_anns, 4)
        ann[:, :, 3] = np.where(np.all(ann == 255, axis=-1), 0, 255).astype(np.uint8)
        ann = self._resize_image(ann, ann.shape[0] // 2)

        duration = 0.5
        num_frames = int(duration * 15)

        frames = [Image.fromarray(bg)]

        for i in list(range(1, num_frames + 1)) + list(range(num_frames, 0, -1)):
            alpha = i / num_frames
            blended_image = self._overlay_images(bg, ann, alpha)
            frames.append(Image.fromarray(blended_image))

        frames[0].save(path, save_all=True, optimize=True, append_images=frames[1:], loop=0)
        # import imageio
        # imageio.mimsave(path, frames)

        sly.logger.info(f"Gif animation saved to: {path}")

    def _overlay_images(self, bg, overlay, opacity):
        alpha = overlay[..., 3] * opacity / 255.0
        one_minus_alpha = 1 - alpha

        result_image = np.copy(bg)
        for c in range(3):
            result_image[..., c] = bg[..., c] * one_minus_alpha + overlay[..., c] * alpha
        return result_image

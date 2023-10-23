import math
import os
import random
from typing import List, Union

import cv2
import numpy as np
import supervisely as sly
from PIL import Image, ImageDraw, ImageFont
from supervisely.imaging import font as sly_font
from tqdm import tqdm

from dataset_tools.image.renders.convert import compress_png

CURENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURENT_DIR))


class Poster:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        force: bool = False,
        is_detection_task: bool = False,
        title: str = None,
        num_classes: int = None,
    ) -> None:
        self.force = force
        self._project_meta = project_meta
        self._api = api if api is not None else sly.Api.from_env()  # ?
        self._project = None
        self._images_info = None
        self._items_count = 0
        self._total_labels = 0
        self._local = False
        self._is_detection_task = is_detection_task

        self._title = title
        self._title_font: str = os.path.join(PARENT_DIR, "fonts/FiraSans-SemiBold.ttf")
        self._subs_font: str = os.path.join(PARENT_DIR, "fonts/FiraSans-Bold.ttf")

        self._num_classes = num_classes

        if isinstance(project, int):
            self._project = self._api.project.get_info_by_id(project)
            self._items_count = self._project.items_count
            for ds in self._api.dataset.get_list(self._project.id):
                for img in self._api.image.get_list(ds.id):
                    self._total_labels += img.labels_count
        elif isinstance(project, str):
            self._local = True
            self._project = sly.Project(project, sly.OpenMode.READ)
            self._items_count = self._project.total_items
            for ds in self._project.datasets:
                ds: sly.Dataset
                img_infos = [
                    ds.get_image_info(sly.fs.get_file_name(ann_name))
                    for ann_name in os.listdir(ds.ann_dir)
                ]
                for img in img_infos:
                    self._total_labels += img.labels_count

        else:
            raise Exception('Parameter "project" has to be one of `int` or `str` types.')

        self._size = (1080, 1960)
        self._GAP = 20
        self._size_line_1 = (
            (self._size[0] - 3 * self._GAP) // 5 * 3,
            (self._size[1] - 4 * self._GAP) // 3,
        )
        self._size_line_2 = (
            (self._size[0] - 3 * self._GAP) // 5 * 2,
            (self._size[1] - 5 * self._GAP) // 4,
        )

        self._poster = np.ones((*self._size, 4), dtype=np.uint8) * 255  # result poster

        self._logo_path = "logo.png"

    @property
    def basename_stem(self) -> None:
        return sly.utils.camel_to_snake(self.__class__.__name__)

    def update(self, data: tuple):
        np_images = []
        join_data = [(ds, img, ann) for ds, list1, list2 in data for img, ann in zip(list1, list2)]
        random.shuffle(join_data)
        i = 0
        with tqdm(desc="Poster: download 7 sample images", total=7) as pbar:
            while len(np_images) < 7:
                if i > len(join_data) * 3:
                    raise Exception("There are not enough images with labels in the project.")
                ds, img_info, ann = join_data[i % len(join_data)]
                ds: sly.Dataset
                if len(ann.labels) < 1:
                    i += 1
                    continue
                np_img = (
                    sly.image.read(ds.get_img_path(img_info.name))
                    if self._local
                    else self._api.image.download_np(img_info.id)
                )

                h, w = np_img.shape[:2]
                background = np.ones((h, w, 3), dtype=np.uint8) * 255

                h, w = self._size_line_1 if i < 3 else self._size_line_2
                img_h, img_w = np_img.shape[:2]
                scale_ratio = max(h / img_h, w / img_w)
                img_h, img_w = int(img_h * scale_ratio), int(img_w * scale_ratio)
                np_img = sly.image.resize(np_img, (img_h, img_w))

                background = sly.image.resize(background, np_img.shape[:2])
                try:
                    ann = ann.resize(np_img.shape[:2])
                except Exception:
                    sly.logger.warn(
                        f"Skipping image: can not resize annotation. Image name: {img_info.name}"
                    )
                    i += 1
                    continue

                ann: sly.Annotation
                thickness = ann._get_thickness()
                for label in ann.labels:
                    if type(label.geometry) == sly.Point:
                        label.draw(np_img, thickness=int(thickness * 2))
                    if self._is_detection_task:
                        bbox = label.geometry.to_bbox()
                        pt1, pt2 = (bbox.left, bbox.top), (bbox.right, bbox.bottom)
                        cv2.rectangle(np_img, pt1, pt2, label.obj_class.color, thickness)
                        font_size = int(sly_font.get_readable_font_size(np_img.shape[:2]) * 1.4)
                        font = sly_font.get_font(font_size=font_size)
                        _, _, _, bottom = font.getbbox(label.obj_class.name)
                        anchor = (bbox.top - bottom, bbox.left)
                        sly.image.draw_text(np_img, label.obj_class.name, anchor, font=font)
                if not self._is_detection_task:
                    ann.draw_pretty(np_img, thickness=3, opacity=0.7, fill_rectangles=False)
                np_img = cv2.addWeighted(np_img, 0.8, background, 0.2, 0)

                # backup
                # if len(np_images) % 2 == 0:
                #     h, w = np_img.shape[:2]
                #     background = np.ones((h, w, 3), dtype=np.uint8) * 255
                #     alpha = 0.5
                #     np_img = cv2.addWeighted(background, 1 - alpha, np_img, alpha, 0)
                #     ann.draw_pretty(np_img, thickness=0, opacity=0.6)
                # else:
                #     ann: sly.Annotation
                #     thickness = 3
                #     ann.draw_contour(np_img, thickness=thickness)

                np_images.append(self._crop_image(np_img, i))
                i += 1
                pbar.update(1)

        self._create_frame(np_images)
        self._draw_text_and_rectangles()

    def update_unlabeled(self, data: tuple):
        np_images = []
        join_data = [(ds, img) for ds, list1, list2 in data for img, ann in zip(list1, list2)]
        random.shuffle(join_data)
        i = 0
        with tqdm(desc="Poster: download 7 sample images", total=7) as pbar:
            while len(np_images) < 7:
                # if i > len(join_data) * 3:
                #     raise Exception("There are not enough images with labels in the project.")
                ds, img_info = join_data[i % len(join_data)]
                ds: sly.Dataset
                # if len(ann.labels) < 1:
                #     i += 1
                #     continue
                np_img = (
                    sly.image.read(ds.get_img_path(img_info.name))
                    if self._local
                    else self._api.image.download_np(img_info.id)
                )

                h, w = np_img.shape[:2]
                background = np.ones((h, w, 3), dtype=np.uint8) * 255

                h, w = self._size_line_1 if i < 3 else self._size_line_2
                img_h, img_w = np_img.shape[:2]
                scale_ratio = max(h / img_h, w / img_w)
                img_h, img_w = int(img_h * scale_ratio), int(img_w * scale_ratio)
                np_img = sly.image.resize(np_img, (img_h, img_w))

                background = sly.image.resize(background, np_img.shape[:2])
                # try:
                #     ann = ann.resize(np_img.shape[:2])
                # except Exception:
                #     sly.logger.warn(
                #         f"Skipping image: can not resize annotation. Image name: {img_info.name}"
                #     )
                #     i += 1
                #     continue

                # ann: sly.Annotation
                # thickness = ann._get_thickness()
                # for label in ann.labels:
                #     if type(label.geometry) == sly.Point:
                #         label.draw(np_img, thickness=int(thickness * 2))
                #     if self._is_detection_task:
                #         bbox = label.geometry.to_bbox()
                #         pt1, pt2 = (bbox.left, bbox.top), (bbox.right, bbox.bottom)
                #         cv2.rectangle(np_img, pt1, pt2, label.obj_class.color, thickness)
                #         font_size = int(sly_font.get_readable_font_size(np_img.shape[:2]) * 1.4)
                #         font = sly_font.get_font(font_size=font_size)
                #         _, _, _, bottom = font.getbbox(label.obj_class.name)
                #         anchor = (bbox.top - bottom, bbox.left)
                #         sly.image.draw_text(np_img, label.obj_class.name, anchor, font=font)
                # if not self._is_detection_task:
                #     ann.draw_pretty(np_img, thickness=3, opacity=0.7, fill_rectangles=False)
                np_img = cv2.addWeighted(np_img, 0.8, background, 0.2, 0)

                # backup
                # if len(np_images) % 2 == 0:
                #     h, w = np_img.shape[:2]
                #     background = np.ones((h, w, 3), dtype=np.uint8) * 255
                #     alpha = 0.5
                #     np_img = cv2.addWeighted(background, 1 - alpha, np_img, alpha, 0)
                #     ann.draw_pretty(np_img, thickness=0, opacity=0.6)
                # else:
                #     ann: sly.Annotation
                #     thickness = 3
                #     ann.draw_contour(np_img, thickness=thickness)

                np_images.append(self._crop_image(np_img, i))
                i += 1
                pbar.update(1)

        self._create_frame(np_images)
        self._draw_text_and_rectangles()

    def to_image(self, path: str = None):
        path_part, ext = os.path.splitext(path)
        tmp_path = f"{path_part}-o{ext}"
        if path is None:
            storage_dir = sly.app.get_data_dir()
            path = os.path.join(storage_dir, "horizontal_grid.png")

        sly.image.write(tmp_path, self._poster)
        compress_png(tmp_path, path)
        sly.fs.silent_remove(tmp_path)
        sly.logger.info(f"Poster saved to: {path}")

    def _crop_image(self, image: np.ndarray, image_num: int):
        h, w = self._size_line_1 if image_num < 3 else self._size_line_2
        img_h, img_w = image.shape[:2]
        start_h = (img_h - h) // 2
        start_w = (img_w - w) // 2
        rect = (start_h, start_w, start_h + h, start_w + w)
        image = sly.image.crop_with_padding(image, sly.Rectangle(*rect))
        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_image[:h, :w, :3] = image[:h, :w, :3]
        return rgba_image

    def _draw_text_and_rectangles(self):
        title, subs = self._get_text_from_project()

        poster_h, poster_w = self._size

        bg_image = np.zeros((poster_h, poster_w, 3), dtype=np.uint8)

        title_bottom = self._draw_title(title)
        image = self._draw_subtitles(subs)
        subs_h, subs_w = image.shape[:2]

        logo = self._draw_logo(subs_h)
        subs_x = (poster_w - subs_w - logo.shape[1]) // 2
        subs_y = title_bottom + self._GAP

        self._gradient(bg_image, subs_x, 0, subs_x + subs_w, poster_h)
        bg = bg_image[subs_y : subs_y + subs_h, subs_x : subs_x + subs_w, :3]

        alpha_channel = np.where(np.all(image == 0, axis=-1), 0, 255).astype(np.uint8)
        text_image_bw = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        _, text_mask = cv2.threshold(text_image_bw, 255, 255, cv2.THRESH_BINARY)
        inverse_text_mask = cv2.bitwise_not(text_mask)
        background_masked = cv2.bitwise_and(bg, bg, mask=inverse_text_mask)
        text_image_alpha = cv2.merge(
            (image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel)
        )
        text_image_alpha = cv2.cvtColor(text_image_alpha, cv2.COLOR_BGRA2BGR)
        result = cv2.add(background_masked, text_image_alpha)
        result = np.hstack([result, logo])

        h, w = result.shape[:2]
        cv2.rectangle(result, (0, 0), (w - 1, h - 1), (255, 255, 255), 5)
        self._poster[subs_y : subs_y + h, subs_x : subs_x + w, :3] = result

    def _draw_title(self, text):
        image_h, image_w = self._size
        font = self._get_base_font_size(self._title_font, text)
        _, top, _, _ = font.getbbox(text)

        full_offset = top
        half_offset = top // 2
        pad = self._GAP // 3
        x_pos_center = int(image_w * 0.5)
        y_pos_percent = int(image_h * 0.5)

        text_width, text_height = font.getsize(text)
        text_color = (0, 0, 0, 210)

        tmp_canvas = np.zeros(
            (text_height + half_offset, text_width + full_offset, 3), dtype=np.uint8
        )
        canvas_h, canvas_w = tmp_canvas.shape[:2]
        tmp_canvas = self._gradient(tmp_canvas, 0, 0, canvas_w, canvas_h)
        tmp_canvas[pad : canvas_h - pad, pad : canvas_w - pad, :3] = 255

        tmp_canvas = Image.fromarray(tmp_canvas)
        draw = ImageDraw.Draw(tmp_canvas)
        text_width, text_height = draw.textsize(text, font=font)
        draw.text((half_offset, -half_offset // 3), text, font=font, fill=text_color)

        tmp_canvas = np.array(tmp_canvas, dtype=np.uint8)
        x, y = (x_pos_center - int(canvas_w / 2), y_pos_percent - int(canvas_h / 2))
        self._poster[y : y + canvas_h, x : x + canvas_w, :3] = tmp_canvas

        return y + canvas_h

    def _get_base_font_size(self, font_family, text):
        image_h, image_w = self._size
        text_width_percent = 90
        text_height_percent = 20
        desired_text_width = image_w * text_width_percent // 100
        desired_text_height = image_h * text_height_percent // 100
        font_size = 30

        font = ImageFont.truetype(font_family, font_size)

        _, _, text_width, text_height = font.getbbox(text)

        while text_width > desired_text_width or text_height > desired_text_height:
            font_size -= 1
            font = font.font_variant(size=font_size)
            text_width, text_height = font.getsize(text)

        desired_font_height = math.ceil((image_h * text_height_percent) // 100)
        desired_font_size = math.ceil(font_size * desired_text_width / text_width)
        desired_font_size = min(desired_font_size, desired_font_height)

        font = font.font_variant(size=desired_font_size)
        return font

    def _draw_subtitles(self, text):
        font_subs = self._get_base_font_size(self._subs_font, text)
        font_subs_for_box = font_subs.font_variant(size=int(font_subs.size * 0.7))
        _, _, _, box_b = font_subs_for_box.getbbox(text)

        font_subs = font_subs.font_variant(size=int(font_subs_for_box.size * 0.6))
        _, _, r, b = font_subs.getbbox(text)
        offset = box_b - b
        image = np.ones((box_b, r + offset, 3), dtype=np.uint8) * 255

        sly.image.draw_text(
            image,
            text,
            (offset // 2, offset // 2),
            font=font_subs,
            fill_background=False,
        )
        image = np.dstack((image, np.ones((*image.shape[:2], 1), dtype=np.uint8) * 255))
        return 255 - image

    def _draw_logo(self, height):
        logo = sly.image.read(self._logo_path)
        logo_h, logo_w = logo.shape[:2]
        scale_factor = height / logo_h
        logo_h, logo_w = height, int(logo_w * scale_factor)

        logo = sly.image.resize(logo, (logo_h, logo_w))
        image = np.zeros((logo_h, logo_w, 4), dtype=np.uint8)
        image[:, :, :3] = logo[:, :, :3]
        return logo

    def _get_text_from_project(self):
        title = None
        if self._title is not None:
            title = self._title.upper()
        else:
            title = self._project.name.upper()

        subs = []
        subs.append(f"{self._items_count} images")
        classes_cnt = self._num_classes or len(self._project_meta.obj_classes)
        classes_text = f'{classes_cnt} {"class" if classes_cnt == 1 else "classes"}'
        subs.append(classes_text)
        if self._total_labels:
            subs.append(f"{self._total_labels} labels")

        return title, " · ".join(subs)

    def _gradient(self, img, left, top, right, bottom):
        c1 = np.array((225, 181, 62))  # rgb
        c2 = np.array((219, 84, 150))  # rgb
        im = np.zeros((bottom - top, right - left, 3), dtype=np.uint8)
        row = np.linspace(0, 1, right - left)
        kernel_1d = np.tile(row, (bottom - top, 1))
        kernel = cv2.merge((kernel_1d, kernel_1d, kernel_1d))
        im = kernel * c1 + (1 - kernel) * c2
        im = im.astype(np.uint8)
        img[top:bottom, left:right, :3] = im
        return img

    def _create_frame(self, images: List[np.ndarray]):
        x, y = self._GAP, self._GAP
        x_end, y_end = x, y

        for img in images[:3]:
            x_end, y_end = x + img.shape[1], y + img.shape[0]
            self._poster[y:y_end, x:x_end] = img
            x = x_end + self._GAP

        y = y_end + self._GAP
        x = self._GAP

        for img in images[3:]:
            x_end, y_end = x + img.shape[1], y + img.shape[0]
            self._poster[y:y_end, x:x_end] = img
            x = x_end + self._GAP

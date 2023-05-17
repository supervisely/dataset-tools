import os
import random
from typing import List, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import supervisely as sly
from supervisely.imaging import font as sly_font


class Poster:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
    ) -> None:
        self._project_meta = project_meta
        self._api = api if api is not None else sly.Api.from_env()  # ?
        self._project = None
        self._images_info = None
        self._items_count = 0
        self._total_labels = 0
        self._local = False

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

        self._size = (540, 960)
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

        self._logo_text = "logo.png"

    def update(self, data: tuple):
        np_images = []
        join_data = [(ds, img, ann) for ds, list1, list2 in data for img, ann in zip(list1, list2)]
        random.shuffle(join_data)
        i = 0
        with tqdm(desc="Downloading 7 sample images.", total=7) as pbar:
            while len(np_images) < 7:
                if i > len(join_data) * 3:
                    raise Exception("There not enought images with labels in the project.")
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

                if len(np_images) % 2 == 0:
                    h, w = np_img.shape[:2]
                    background = np.ones((h, w, 3), dtype=np.uint8) * 255
                    alpha = 0.5
                    np_img = cv2.addWeighted(background, 1 - alpha, np_img, alpha, 0)
                    ann.draw_pretty(np_img, thickness=0, opacity=0.3)
                else:
                    thickness = 7 if len(np_images) < 3 else 9
                    ann.draw_contour(np_img, color=[253, 69, 133], thickness=thickness)

                np_images.append(np_img)
                i += 1
                pbar.update(1)

        self._resize_images(np_images)
        self._create_frame(np_images)
        self._draw_text_and_bboxes()
        self._put_watermark(self._logo_text)

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            sly.fs.clean_dir(storage_dir)
            path = os.path.join(storage_dir, "poster.jpeg")
        sly.image.write(path, self._poster)
        sly.logger.info(f"Poster saved to: {path}")

    def _resize_images(self, images):
        for i, img in enumerate(images):
            h, w = (self._size_line_1) if i < 3 else (self._size_line_2)
            img_h, img_w = img.shape[1], img.shape[0]
            src_ratio = w / h
            img_ratio = img_w / img_h
            if img_ratio != src_ratio:
                if img_ratio > src_ratio:
                    img_h, img_w = int(w * img_ratio), w
                    img = sly.image.resize_inter_nearest(img, (img_h, img_w))
                else:
                    img_h, img_w = h, int(h / img_ratio)
                    img = sly.image.resize_inter_nearest(img, (img_h, img_w))
                rect = ((img_h - h) // 2, (img_w - w) // 2, (img_h + h) // 2, (img_w + w) // 2)
                img = sly.image.crop_with_padding(img, sly.Rectangle(*rect))
            else:
                img = sly.image.resize(img, (h, w))

            rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = img
            images[i] = rgba_image
        sly.logger.info(f"Resized {len(images)} sample images.")

    def _draw_text_and_bboxes(self):
        base_font_size = sly_font.get_readable_font_size(self._size)
        fonts = [
            sly_font.get_font(font_size=int(base_font_size * 6)),  # title
            sly_font.get_font(font_size=int(base_font_size * 1.5)),  # images count
            sly_font.get_font(font_size=int(base_font_size * 1.3)),  # classes count
            sly_font.get_font(font_size=int(base_font_size * 1)),  # labels count
        ]
        texts = self._get_text_from_project()

        h, w = self._size

        bg_image = np.zeros((h, w, 3), dtype=np.uint8)

        def _get_text_bbox(text, font):
            if text is None:
                return None
            left, top, right, bottom = font.getbbox(text)
            text_w, text_h = right - left, bottom
            while text_w > w * 0.9:
                font_size *= 0.96
                left, top, right, bottom = font.getbbox(text)
                text_w, text_h = right - left, bottom
            return text_w, text_h

        text_sizes = [_get_text_bbox(t, f) for t, f in zip(texts, fonts)]

        x1, y1 = (w - text_sizes[0][0]) // 2, (h - text_sizes[0][1]) // 2
        text_coords = [
            (x1, y1),
            (x1, (y1 - text_sizes[1][1])),
            (x1 + text_sizes[0][0] - text_sizes[2][0], (y1 - text_sizes[2][1])),
        ]
        if text_sizes[3] is not None:
            text_coords.append((w // 2 - text_sizes[3][0], (y1 + text_sizes[0][1])))

        bg_image = self._gradient(bg_image, x1, 0, x1 + text_sizes[0][0], h)

        for idx, (t, c, s, f) in enumerate(zip(texts, text_coords, text_sizes, fonts)):
            if c is False:
                continue
            left, top, right, bottom = c[0], c[1], c[0] + s[0], c[1] + s[1]
            if idx == 0:
                offset = self._GAP // 5
                bg_image[top + offset : bottom - offset, left + offset : right - offset] = 255
            offs_x = (right - left - f.getlength(t)) // 2
            offs_y = f.getbbox(t)[1] // 2
            sly.image.draw_text(
                bg_image, t, (top - offs_y, left + offs_x), font=f, fill_background=False
            )

            self._poster[top:bottom, left:right, :3] = bg_image[top:bottom, left:right, :3]

    def _get_text_from_project(self):
        title = (
            self._project.name.upper()
            if len(self._project.name.split(" ")) < 4
            else " ".join(self._project.name.split(" ")[:3]).upper()
        )
        images_text = f"{self._items_count} IMAGES"
        classes_text = f"{len(self._project_meta.obj_classes)} CLASSES"
        labels_text = f"{self._total_labels} LABELS" if self._total_labels else None

        return title, images_text, classes_text, labels_text

    def _gradient(self, img, left, top, right, bottom):
        c1 = np.array((210, 95, 144))  # rgb (253, 69, 133)
        c2 = np.array((234, 197, 77))  # (254, 208, 0)
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

    def _put_watermark(self, path):
        img = sly.image.read(path)
        poster_h, poster_w = self._size

        # resize logo to poster size
        mark_w = int(self._size_line_2[0] * 0.9)
        mark_h = mark_w * img.shape[0] // img.shape[1]
        img = cv2.resize(img, (mark_w, mark_h), interpolation=cv2.INTER_NEAREST)
        # img = sly.image.resize_inter_nearest(img, (mark_h, mark_w))
        img_h, img_w = img.shape[:2]

        # calculate coords
        offset_x = (self._size_line_2[1] - img_w) // 2
        x = 3 * self._size_line_2[1] + 4 * self._GAP + offset_x
        y = int(poster_h - 1.2 * self._GAP - img_h)

        # add logo
        logo = np.zeros((poster_h, poster_w, 4), dtype=np.uint8)
        logo[y : y + img_h, x : x + img.shape[1], :3] = img[:, :, :3]
        logo[y : y + img_h, x : x + img.shape[1], 3] = 255

        self._poster = cv2.addWeighted(self._poster, 0.9, logo, 0.5, 0)

        # or
        # background = Image.fromarray(self._poster, mode="RGBA")
        # overlay = Image.fromarray(logo, mode="RGBA")
        # overlay = overlay.resize(background.size)
        # new_image = Image.blend(background, overlay, alpha=0.15)
        # self._poster = np.array(new_image, dtype=np.uint8)

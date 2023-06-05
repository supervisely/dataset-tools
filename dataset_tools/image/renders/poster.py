import os
import random
from typing import List, Union

import cv2
import numpy as np
from tqdm import tqdm

import supervisely as sly
from dataset_tools.image.renders.convert import compress_png
from supervisely.imaging import font as sly_font


class Poster:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        force: bool = False,
    ) -> None:
        self.force = force
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

                h, w = np_img.shape[:2]
                background = np.ones((h, w, 3), dtype=np.uint8) * 255
                ann.draw_pretty(np_img, thickness=0, opacity=0.7)
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

                np_images.append(self._resize_image(np_img, i))
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

    def _resize_image(self, image: np.ndarray, image_num: int):
        h, w = self._size_line_1 if image_num < 3 else self._size_line_2
        img_h, img_w = image.shape[:2]
        scale_ratio = max(h / img_h, w / img_w)
        img_h, img_w = int(img_h * scale_ratio), int(img_w * scale_ratio)
        image = sly.image.resize(image, (img_h, img_w))
        start_h = (img_h - h) // 2
        start_w = (img_w - w) // 2
        rect = (start_h, start_w, start_h + h, start_w + w)
        image = sly.image.crop_with_padding(image, sly.Rectangle(*rect))

        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_image[:h, :w, :3] = image[:h, :w, :3]
        return rgba_image

    def _draw_text_and_rectangles(self):
        base_font_size = sly_font.get_readable_font_size(self._size)
        font_name_title = "FiraSans-Regular.ttf"
        font_name_subs = "FiraSans-Thin.ttf"
        font_title = sly_font.get_font(font_name_title, int(base_font_size * 6))
        font_subs = sly_font.get_font(font_name_subs, int(base_font_size * 1.8))
        title, *subs = self._get_text_from_project()
        _, _, _, subs_height = font_subs.getbbox("".join(subs))

        poster_h, poster_w = self._size

        bg_image = np.zeros((poster_h, poster_w, 3), dtype=np.uint8)

        title_bottom = self._draw_title(bg_image, font_title, title)
        sub_imgs = [self._draw_subtitles(font_subs, text, subs_height) for text in subs]
        logo = self._draw_logo(subs_height)

        image = np.hstack(sub_imgs)
        subs_h, subs_w = image.shape[:2]
        subs_x = (poster_w - subs_w - logo.shape[1]) // 2
        subs_y = title_bottom + self._GAP

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
        cv2.rectangle(result, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
        self._poster[subs_y : subs_y + h, subs_x : subs_x + w, :3] = result

    def _draw_title(self, image, font, text):
        poster_h, poster_w = self._size
        l, t, r, b = font.getbbox(text)
        title_w, title_h = r - l, b
        while title_w > poster_w * 0.9:
            font = font.font_variant(size=int(font.size * 0.96))
            l, t, r, b = font.getbbox(text)
            title_w, title_h = r - l, b
        title_x, title_y = (poster_w - title_w) // 2, (poster_h - title_h) // 2
        left, top, right, bottom = title_x, title_y, title_x + title_w, title_y + title_h

        self._gradient(image, title_x, 0, title_x + title_w, poster_h)
        p = self._GAP // 3
        image[top + p : bottom - p, left + p : right - p] = 255

        sly.image.draw_text(
            image,
            text,
            (top - t // 2, left),
            font=font,
            fill_background=False,
            color=(0, 0, 0, 210),
        )

        self._poster[top:bottom, left:right, :3] = image[top:bottom, left:right, :3]
        return bottom + p

    def _draw_subtitles(self, font, text, height):
        _, t, r, _ = font.getbbox(text)
        pad = self._GAP // 4
        w = r + self._GAP
        image = np.ones((height, w, 3), dtype=np.uint8) * 255
        sly.image.draw_text(
            image,
            text,
            (-t // 2, pad * 2),
            font=font,
            fill_background=False,
        )
        image[:, r + 3 * pad :, :3] = 0
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
        texts = []
        texts.append(
            self._project.name.upper()
            if len(self._project.name.split(" ")) < 4
            else " ".join(self._project.name.split(" ")[:3]).upper()
        )

        texts.append(f"{self._items_count} IMAGES")
        classes_cnt = len(self._project_meta.obj_classes)
        classes_text = f'{classes_cnt} {"CLASS" if classes_cnt == 1 else "CLASSES"}'
        texts.append(classes_text)
        if self._total_labels:
            texts.append(f"{self._total_labels} LABELS")

        return texts

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

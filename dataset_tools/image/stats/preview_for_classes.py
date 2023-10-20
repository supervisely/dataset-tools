import gc
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Union

import cv2
import numpy as np
import supervisely as sly
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from dataset_tools.image.renders.convert import compress_mp4, from_mp4_to_webm
from dataset_tools.image.stats.basestats import BaseVisual

UNLABELED_COLOR = [0, 0, 0]
GRADIEN_COLOR_1 = (225, 181, 62)
GRADIEN_COLOR_2 = (219, 84, 150)
font_name = "FiraSans-Regular.ttf"

CLASSES_CNT_LIMIT = 25
LABELAREA_THRESHOLD_SMALL = 200 * 200
LABELAREA_THRESHOLD_BIG = 400 * 400

CURENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURENT_DIR))


        
class ClassesPreview(BaseVisual):
    """Get previews for classes in the dataset."""

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_info: sly.ProjectInfo,
        api: sly.Api = None,
        row_height: int = None,
        force: bool = False,
        pad: dict = {"top": "10%", "bottom": "10%", "left": "10%", "right": "10%"},
        rows: int = None,
        gap: int = 20,
    ):
        self.force = force
        self._meta = project_meta
        self._project_name = project_info.name
        classes_cnt = len(self._meta.obj_classes)
        classes_text = "classes" if classes_cnt > 1 else "class"
        self._title = f"{self._project_name} · {classes_cnt} {classes_text} "
        self._pad = pad

        self._gap = gap
        self._rows = rows
        self._first_row_height = None
        self._last_row_height = None
        self._img_width = 1920
        self._img_height = None
        self._row_height = row_height if row_height is not None else 480
        self._row_width = None
        self._font = None

        self._api = api if api is not None else sly.Api.from_env()

        self._classname2images = defaultdict(list)
        self._bigclasses = defaultdict(list)
        self._np_images = {}
        self._np_anns = {}
        self._np_texts = {}

        self._logo_path = "logo.png"

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        for label in ann.labels:
            class_name = label.obj_class.name
            label_bbox = label.geometry.to_bbox()
            if (
                (label_bbox.area < LABELAREA_THRESHOLD_SMALL)
            ):
                continue
            if (
                (label_bbox.area > LABELAREA_THRESHOLD_BIG)
            ):
                self._bigclasses[class_name].append(image.id)
            if image.id not in self._classname2images[class_name]:
                self._classname2images[class_name].append(image.id)

    def animate(
        self,
        path: str = None,
        font: str = os.path.join(PARENT_DIR, "fonts/FiraSans-Bold.ttf"),
    ):
        if len(self._classname2images) > 0:
            self._font = font
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
            self._collect_images()
            canvas, masks, texts = self._prepare_layouts()

            duration, fps = 1.5, 15
            num_frames = int(duration * fps)
            frames = []
            num_frames_list = [0] * 10
            num_frames_list.extend(list(range(0, num_frames)))
            num_frames_list.extend([num_frames] * 10)
            num_frames_list.extend(list(range(num_frames, 0, -1)))

            for i in num_frames_list:
                alpha = i / num_frames
                if i == num_frames:
                    alpha = round(1 - random.uniform(0, 0.03), 3)
                elif i == 0:
                    alpha = round(0 + random.uniform(0, 0.03), 3)
                frame = self._overlay_images(canvas, masks, alpha)
                frame = self._overlay_images(frame, texts, 1)
                frame = self._add_logo(frame)
                frame = self._draw_title(frame, self._title)
                frames.append(frame)
        else:
            text = "No suitable images to create a classes preview."
            sly.logger.warn(text)
            frames = [self._create_empty_frame(font, text)]

        tmp_video_path = f"{os.path.splitext(path)[0]}-o.mp4"
        video_path = f"{os.path.splitext(path)[0]}.mp4"
        self._save_video(tmp_video_path, frames)
        from_mp4_to_webm(tmp_video_path, path)
        compress_mp4(tmp_video_path, video_path)
        sly.fs.silent_remove(tmp_video_path)

        sly.logger.info(f"Animation saved to: {path}, {video_path}")
        self._classname2images = None
        self._np_images = None
        self._np_anns = {}

    def _prepare_layouts(self):
        canvas = self._create_grid(list(self._np_images.values()))
        masks = self._create_grid(list(self._np_anns.values()))
        texts = self._create_grid(list(self._np_texts.values()))

        alpha = np.zeros((self._img_height, self._row_width), dtype=canvas.dtype)
        canvas = cv2.merge([canvas, alpha])
        masks = cv2.merge([masks, alpha])
        texts = cv2.merge([texts, alpha])
        masks[:, :, 3] = np.where(np.all(masks == 0, axis=-1), 0, 255).astype(np.uint8)
        texts[:, :, 3] = np.where(np.all(texts == 0, axis=-1), 0, 255).astype(np.uint8)

        return canvas, masks, texts

    def _collect_images(self) -> None:
        classes_cnt = len(self._classname2images)
        limit = classes_cnt if classes_cnt < CLASSES_CNT_LIMIT else CLASSES_CNT_LIMIT
        i = 0
        for big_class in self._bigclasses:
            self._classname2images[big_class] = self._bigclasses[big_class]
        with tqdm(
            desc="ClassesPreview: download and prepare images with annotations",
            total=limit,
        ) as pbar:
            while len(self._np_images) < limit:
                cls_name, ids = list(self._classname2images.items())[i]
                random.shuffle(ids)
                image_id = ids[0]
                img = self._api.image.download_np(image_id)
                ann_json = self._api.annotation.download_json(image_id)
                ann = sly.Annotation.from_json(ann_json, self._meta)

                grouped_labels = {}
                for label in ann.labels:
                    if label.obj_class.name in grouped_labels:
                        grouped_labels[label.obj_class.name].append(label)
                    else:
                        grouped_labels[label.obj_class.name] = [label]
                cls_type = self._meta.get_obj_class(cls_name).geometry_type
                if cls_type != sly.Rectangle:
                    for key, labels in grouped_labels.items():
                        if len(labels)>1:
                            grouped_labels[key] = [label for label in labels if not isinstance(label.geometry, sly.Rectangle)]

                refined_labels_flat  = [value for values in grouped_labels.values() for value in values]
                ann = ann.clone(labels=refined_labels_flat)

                if not self._bigclasses[cls_name]:
                    pad = {"top": "15%", "bottom": "15%", "left": "15%", "right": "15%"}
                    crop_threshold = LABELAREA_THRESHOLD_SMALL
                else:
                    pad = self._pad
                    crop_threshold = LABELAREA_THRESHOLD_BIG

                crops = sly.aug.instance_crop(
                    img=img,
                    ann=ann,
                    class_title=cls_name,
                    save_other_classes_in_crop=False,
                    padding_config=pad,
                )
                random.shuffle(crops)
                for crop in crops:
                    if crop[1].img_size[0]*crop[1].img_size[1] > crop_threshold:
                        cropped_img, cropped_ann = crop
                        break
                cropped_img = self._resize_image_by_height(cropped_img, self._row_height)
                try:
                    cropped_ann = cropped_ann.resize(cropped_img.shape[:2])
                except Exception:
                    sly.logger.warn(
                        f"Skipping image: can not resize annotation. Image id: {image_id}"
                    )
                    i += 1
                    continue
                ann_mask = np.zeros((*cropped_img.shape[:2], 3), dtype=np.uint8)
                text_mask = np.zeros((*cropped_img.shape[:2], 3), dtype=np.uint8)

                for label in cropped_ann.labels:
                    if type(label.geometry) == sly.Rectangle:
                        label.draw_contour(ann_mask, thickness=5)
                    else:
                        label.draw(ann_mask, thickness=5)
                    font_size = self._get_base_font_size(cls_name, ann_mask.shape[:2])
                    font = ImageFont.truetype(self._font, int(font_size * 0.75))

                    white = (255, 255, 255, 255)
                    black = (0, 0, 0, 0)
                    x, y = cropped_img.shape[1]/2, cropped_img.shape[0]/2
                    tmp_canvas = Image.fromarray(text_mask)
                    draw = ImageDraw.Draw(tmp_canvas)
                    draw.text(
                        (x, y), cls_name, white, font, "mm", stroke_width=1, stroke_fill=black
                    )
                    text_mask = np.array(tmp_canvas, dtype=np.uint8)

                self._np_images[cls_name] = cropped_img
                self._np_anns[cls_name] = ann_mask
                self._np_texts[cls_name] = text_mask
                i += 1
                pbar.update(1)

    def _resize_image_by_height(self, image: np.ndarray, height: int) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = height / img_h
        img_h = height
        img_w = int(img_aspect_ratio * img_w)
        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _create_grid(self, images: List[np.ndarray]) -> np.ndarray:
        channels = images[0].shape[-1]
        rows = self._create_rows(images)
        rows = self._merge_img_in_rows(rows, channels)
        canvas = self._merge_rows_in_canvas(rows, channels)

        return canvas

    def _create_rows(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        images = sorted(images, key=lambda x: x.shape[1], reverse=True)

        def _split_list_images(images, n):
            rows = [[] for _ in range(n)]
            sums = [self._gap] * n
            total_height = n * self._row_height + self._gap * (n + 1)

            if len(images) == 1 and n > 1:
                single_width = images[0].shape[1] + self._gap * 2
                return [images], abs(single_width - total_height)

            for image in images:
                min_sum_index = sums.index(min(sums))
                rows[min_sum_index].append(image)
                sums[min_sum_index] += image.shape[1] + self._gap

            aspect_diff = abs(max(sums) - total_height)
            return rows, aspect_diff

        if self._rows is not None:
            rows, _ = _split_list_images(images, self._rows)
            return rows
        row_cnt = 1
        rows, diff = _split_list_images(images, row_cnt)
        while True:
            row_cnt += 1
            cur_rows, cur_diff = _split_list_images(images, row_cnt)
            if cur_diff >= diff * 0.7 or row_cnt > len(images):
                row_cnt -= 1
                break
            rows, diff = cur_rows, cur_diff
        self._rows = row_cnt
        return rows

    def _merge_img_in_rows(self, rows: List[np.ndarray], channels: int = 3) -> List[np.ndarray]:
        row_widths = []
        img_height = self._gap
        for row in rows:
            sum_widths = sum([img.shape[1] for img in row]) + self._gap * (len(row) + 1)
            row_widths.append(sum_widths)
        avg_row_width = sorted(row_widths)[len(row_widths) // 2]
        avg_row_width = min(avg_row_width, self._img_width)

        combined_rows = []
        for row in rows:
            if len(row) == 0:
                continue

            separator = np.ones((self._row_height, self._gap, channels), dtype=np.uint8) * 255
            combined_images = []

            for image in row[::-1]:
                combined_images.append(image)
                combined_images.append(separator)
            combined_images.pop()
            combined_image = np.hstack(combined_images)
            SCALE_TRIGGER = 0.7
            if combined_image.shape[1] > avg_row_width * SCALE_TRIGGER:
                aspect_ratio = avg_row_width / combined_image.shape[1]
                img_h = int(aspect_ratio * combined_image.shape[0])
                combined_image = self._resize_image_by_height(combined_image, img_h)
            img_height += combined_image.shape[0] + self._gap
            combined_rows.append(combined_image)

        self._row_width = avg_row_width
        self._img_height = img_height
        self._first_row_height = combined_rows[0].shape[0]
        self._last_row_height = combined_rows[-1].shape[0]
        return combined_rows

    def _merge_rows_in_canvas(self, rows, channels: int = 3) -> np.ndarray:
        canvas = np.ones([self._img_height, self._row_width, channels], dtype=np.uint8) * 255
        y = self._gap
        for image in rows:
            if image.shape[1] > canvas.shape[1] - self._gap * 2:
                image = image[:, : canvas.shape[1] - self._gap * 2]

            row_start = y
            row_end = row_start + image.shape[0]
            column_start = self._gap
            column_end = image.shape[1] + self._gap
            y = row_end + self._gap

            if row_end > canvas.shape[0]:
                continue
            canvas[row_start:row_end, column_start:column_end, :channels] = image[:, :, :channels]

        return canvas

    def _overlay_images(
        self, bg: np.ndarray, overlay: np.ndarray, opacity: Union[float, int]
    ) -> np.ndarray:
        alpha = overlay[..., 3] * opacity / 255.0
        one_minus_alpha = 1 - alpha

        result_image = np.copy(bg)
        for c in range(3):
            result_image[..., c] = bg[..., c] * one_minus_alpha + overlay[..., c] * alpha
        return result_image

    def _add_logo(self, image):
        height = self._last_row_height // 5
        image2 = cv2.imread(self._logo_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGBA)
        image2 = self._resize_image_by_height(image2, height)

        height_r = height
        while image2.shape[1] > image.shape[1] * 0.25:
            height_r = int(0.95 * height_r)
            image2 = self._resize_image_by_height(image2, height_r)

        h1, w1 = image.shape[:2]
        h2, w2 = image2.shape[:2]
        x = w1 - w2 - self._gap * 2
        y = h1 - h2 - self._gap * 2
        alpha = image2[:, :, 3] / 255.0

        reg = image[y : y + h2, x : x + w2]
        image[y : y + h2, x : x + w2, :3] = (1 - alpha[:, :, np.newaxis]) * reg[:, :, :3] + alpha[
            :, :, np.newaxis
        ] * image2[:, :, :3]

        return image

    def _draw_title(self, image, text):
        _, image_w = image.shape[:2]
        font_size = self._get_base_font_size(text, (self._first_row_height, image_w))
        font = ImageFont.truetype(self._font, int(font_size * 0.75))
        _, top, _, _ = font.getbbox(text)

        full_offset = top * 2
        half_offset = top
        x_pos_center = int(image_w * 0.5)
        y_pos_percent = self._gap * 2

        text_width, text_height = font.getsize(text)
        text_color = (255, 255, 255, 255)

        tmp_canvas = np.zeros(
            (text_height + half_offset, text_width + full_offset, 3), dtype=np.uint8
        )
        tmp_canvas = self._gradient(
            tmp_canvas, 0, 0, text_width + full_offset, text_height + half_offset
        )

        tmp_canvas = Image.fromarray(tmp_canvas)
        canvas_w, canvas_h = tmp_canvas.size
        draw = ImageDraw.Draw(tmp_canvas)
        text_width, text_height = draw.textsize(text, font=font)
        x, y = (x_pos_center - int(text_width / 2), y_pos_percent)
        draw.text((canvas_w // 2, canvas_h // 2), text, text_color, font, "mm")

        tmp_canvas = np.array(tmp_canvas, dtype=np.uint8)

        image[y - 3 : y + canvas_h + 3, x - 3 : x + canvas_w + 3, :3] = 255
        image[y : y + canvas_h, x : x + canvas_w, :3] = tmp_canvas

        return image

    def _get_base_font_size(self, text: str, size: Tuple[int, int]):
        image_h, image_w = size
        desired_text_width = image_w * 0.85
        desired_text_height = image_h * 0.2
        text_height_percent = 25
        font_size = 30

        font = ImageFont.truetype(self._font, font_size)

        text_width, text_height = font.getsize(text)

        while text_width > desired_text_width or text_height > desired_text_height:
            font_size -= 1
            if font_size < 1:
                break
            font = ImageFont.truetype(self._font, font_size)
            text_width, text_height = font.getsize(text)

        desired_font_height = math.ceil((image_h * text_height_percent) // 100)
        desired_font_size = math.ceil(font_size * desired_text_width / text_width)
        desired_font_size = min(desired_font_size, desired_font_height)

        return desired_font_size

    def _gradient(self, img, left, top, right, bottom):
        c1 = np.array(GRADIEN_COLOR_1)  # rgb
        c2 = np.array(GRADIEN_COLOR_2)  # rgb
        im = np.zeros((bottom - top, right - left, 3), dtype=np.uint8)
        row = np.linspace(0, 1, right - left)
        kernel_1d = np.tile(row, (bottom - top, 1))
        kernel = cv2.merge((kernel_1d, kernel_1d, kernel_1d))
        im = kernel * c1 + (1 - kernel) * c2
        im = im.astype(np.uint8)
        img[top:bottom, left:right, :3] = im
        return img

    def _save_video(self, videopath: str, frames):
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
        height, width = frames[0].shape[:2]
        video_writer = cv2.VideoWriter(videopath, fourcc, 15, (width, height))

        with tqdm(desc="ClassesPreview: saving video...", total=len(frames)) as vid_pbar:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                video_writer.write(frame)
                vid_pbar.update(1)
        video_writer.release()

    def _create_empty_frame(self, font, text):
        self._font = font
        frame = np.zeros((self._row_height, self._img_width, 3), dtype=np.uint8)
        font_size = self._get_base_font_size(text, frame.shape[:2])
        font = ImageFont.truetype(font, int(font_size * 0.75))
        white = (255, 255, 255, 255)
        tmp_canvas = Image.fromarray(frame)
        draw = ImageDraw.Draw(tmp_canvas)
        anchor = (frame.shape[1] // 2, frame.shape[0] // 2)
        draw.text(anchor, text, white, font, "mm")
        frame = np.array(tmp_canvas, dtype=np.uint8)
        return frame


class ClassesPreviewTags(ClassesPreview):
    """Get previews for classes in the dataset."""

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_info: sly.ProjectInfo,
        tags: List[str] = None,        
        api: sly.Api = None,
        row_height: int = None,
        force: bool = False,
        pad: dict = {"top": "10%", "bottom": "10%", "left": "10%", "right": "10%"},
        rows: int = None,
        gap: int = 20,
    ):
        super().__init__(project_meta, project_info, api, row_height, force, pad, rows, gap)
        self._tags = tags
        if self._tags is not None:
            classes_cnt = len(self._tags)
            classes_text = "classes" if classes_cnt > 1 else "class"
            self._title = f"{self._project_name} · {classes_cnt} {classes_text} "

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:              
        for class_tag in ann.img_tags.items():
            if class_tag.name in self._tags:                
                self._classname2images[class_tag.name].append(image.id) 

    def animate(
        self,
        path: str = None,
        font: str = os.path.join(PARENT_DIR, "fonts/FiraSans-Bold.ttf"),
    ):
        if len(self._classname2images) > 0:
            self._font = font
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
            self._collect_images()
            canvas, texts = self._prepare_layouts()

            # duration, fps = 1.5, 15
            # num_frames = int(duration * fps)
            num_frames = 1
            frames = []
            num_frames_list = [0] * 10
            num_frames_list.extend(list(range(0, num_frames)))
            num_frames_list.extend([num_frames] * 10)
            num_frames_list.extend(list(range(num_frames, 0, -1)))

            for _ in num_frames_list:
                frame = self._overlay_images(canvas, texts, 1)
                frame = self._add_logo(frame)
                frame = self._draw_title(frame, self._title)
                frames.append(frame)
        else:
            text = "No suitable images to create a classes preview."
            sly.logger.warn(text)
            frames = [self._create_empty_frame(font, text)]

        tmp_video_path = f"{os.path.splitext(path)[0]}-o.mp4"
        video_path = f"{os.path.splitext(path)[0]}.mp4"
        self._save_video(tmp_video_path, frames)
        from_mp4_to_webm(tmp_video_path, path)
        compress_mp4(tmp_video_path, video_path)
        sly.fs.silent_remove(tmp_video_path)

        sly.logger.info(f"Animation saved to: {path}, {video_path}")
        self._classname2images = None
        self._np_images = None
        self._np_anns = {}

    def _collect_images(self) -> None:
        classes_cnt = len(self._classname2images)
        limit = classes_cnt if classes_cnt < CLASSES_CNT_LIMIT else CLASSES_CNT_LIMIT
        i = 0

        with tqdm(
            desc="ClassesPreviewTags: download and prepare images without annotations",
            total=limit,
        ) as pbar:
            while len(self._np_images) < limit:
                cls_name, ids = list(self._classname2images.items())[i]
                random.shuffle(ids)
                image_id = ids[0]
                img = self._api.image.download_np(image_id)
                ann_json = self._api.annotation.download_json(image_id)
                ann = sly.Annotation.from_json(ann_json, self._meta)

                img = self._resize_image_by_height(img, self._row_height)
                text_mask = np.zeros((*img.shape[:2], 3), dtype=np.uint8)

                for class_tag in ann.img_tags.items():
                    font_size = self._get_base_font_size(cls_name, img.shape[:2])
                    font = ImageFont.truetype(self._font, int(font_size * 0.75))

                    white = (255, 255, 255, 255)
                    black = (0, 0, 0, 0)
                    x, y = img.shape[1]/2, img.shape[0]/2
                    tmp_canvas = Image.fromarray(text_mask)
                    draw = ImageDraw.Draw(tmp_canvas)
                    draw.text(
                        (x, y), cls_name, white, font, "mm", stroke_width=1, stroke_fill=black
                    )
                    text_mask = np.array(tmp_canvas, dtype=np.uint8)

                self._np_images[cls_name] = img                
                self._np_texts[cls_name] = text_mask
                i += 1
                pbar.update(1)

    def _prepare_layouts(self):
        canvas = self._create_grid(list(self._np_images.values()))
        texts = self._create_grid(list(self._np_texts.values()))

        alpha = np.zeros((self._img_height, self._row_width), dtype=canvas.dtype)
        canvas = cv2.merge([canvas, alpha])
        texts = cv2.merge([texts, alpha])
        texts[:, :, 3] = np.where(np.all(texts == 0, axis=-1), 0, 255).astype(np.uint8)

        return canvas, texts                
    

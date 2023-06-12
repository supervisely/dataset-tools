import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Union

import cv2
import numpy as np
import supervisely as sly
from PIL import Image, ImageDraw, ImageFont
from supervisely.imaging import font as sly_font
from tqdm import tqdm

from dataset_tools.image.renders.convert import compress_mp4, from_mp4_to_webm
from dataset_tools.image.stats.basestats import BaseVisual

UNLABELED_COLOR = [0, 0, 0]
font_name = "FiraSans-Regular.ttf"

CURENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURENT_DIR)))


class ClassesPreview(BaseVisual):
    """Get previews for classes in the dataset."""

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_name: str,
        api: sly.Api = None,
        row_height: int = None,
        force: bool = False,
        pad: dict = {"top": "10%", "bottom": "10%", "left": "10%", "right": "10%"},
    ):
        self.force = force
        self._meta = project_meta
        self._project_name = project_name
        classes_cnt = len(self._meta.obj_classes)
        classes_text = "classes" if classes_cnt > 1 else "class"
        self._title = f"{self._project_name} · {classes_cnt} {classes_text}"
        self._pad = pad

        self._gap = 20
        self._img_width = 1920
        self._img_height = None
        self._row_height = row_height if row_height is not None else 480
        self._row_width = None
        self._font = None

        self._api = api if api is not None else sly.Api.from_env()

        self._classname2images = defaultdict(list)
        self._np_images = {}
        self._np_anns = {}
        self._np_texts = {}

        self._logo_path = "logo.png"

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        for label in ann.labels:
            image_area = image.width * image.height
            if image_area * 0.8 < label.area < image_area * 0.1:
                continue
            class_name = label.obj_class.name
            self._classname2images[class_name].append((image, ann))

    def animate(
        self,
        path: str = None,
        font: str = os.path.join(PARENT_DIR, "fonts/FiraSans-Bold.ttf"),
    ):
        self._font = font
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        self._collect_images()
        canvas, masks, texts = self._prepare_layouts()

        duration, fps = 1.1, 15
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

        tmp_video_path = f"{os.path.splitext(path)[0]}-o.mp4"
        video_path = f"{os.path.splitext(path)[0]}.mp4"
        self._save_video(tmp_video_path, frames)
        from_mp4_to_webm(tmp_video_path, path)
        compress_mp4(tmp_video_path, video_path)
        sly.fs.silent_remove(tmp_video_path)

        sly.logger.info(f"Animation saved to: {path}, {video_path}")

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
        with tqdm(
            desc="Download and prepare images and annotations",
            total=len(self._classname2images),
        ) as pbar:
            for cls_name, items in self._classname2images.items():
                random.shuffle(items)
                items: List[Tuple[sly.ImageInfo, sly.Annotation]]
                items = sorted(
                    items,
                    key=lambda item: max(
                        [
                            label.area / (item[0].width * item[0].height)
                            for label in item[1].labels
                            if label.obj_class.name == cls_name
                        ]
                    ),
                )
                image, ann = items[len(items) // 2]

                image_area = image.width * image.height
                img = self._api.image.download_np(image.id, keep_alpha=True)

                crops = sly.aug.instance_crop(
                    img=img,
                    ann=ann,
                    class_title=cls_name,
                    save_other_classes_in_crop=False,
                    padding_config=self._pad,
                )
                crops = sorted(crops, key=lambda crop: crop[1].labels[0].area / image_area)
                cropped_img, cropped_ann = crops[-1]

                cropped_img = self._resize_image(cropped_img, self._row_height)
                cropped_ann = cropped_ann.resize(cropped_img.shape[:2])
                ann_mask = np.zeros((*cropped_img.shape[:2], 3), dtype=np.uint8)
                text_mask = np.zeros((*cropped_img.shape[:2], 3), dtype=np.uint8)

                for label in cropped_ann.labels:
                    if type(label.geometry) == sly.Rectangle:
                        label.draw_contour(ann_mask, thickness=5)
                    else:
                        label.draw(ann_mask, thickness=5)

                    bbox = label.geometry.to_bbox()

                    font_size = self._get_base_font_size(cls_name, ann_mask.shape[1])
                    font = ImageFont.truetype(self._font, font_size // 2)

                    color_white = (255, 255, 255, 255)
                    color_black = (0, 0, 0, 0)
                    x_pos, y_pos = bbox.center.col, bbox.center.row

                    tmp_canvas = Image.fromarray(text_mask)
                    draw = ImageDraw.Draw(tmp_canvas)
                    draw.text(
                        (x_pos, y_pos),
                        cls_name,
                        font=font,
                        fill=color_white,
                        stroke_width=1,
                        stroke_fill=color_black,
                        anchor="mm",
                    )
                    text_mask = np.array(tmp_canvas, dtype=np.uint8)

                self._np_images[cls_name] = cropped_img
                self._np_anns[cls_name] = ann_mask
                self._np_texts[cls_name] = text_mask
                pbar.update(1)

    def _resize_image(self, image: np.ndarray, height: int) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = height / img_h
        img_h = height
        img_w = int(img_aspect_ratio * img_w)
        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _create_grid(self, images: List[np.ndarray], channels: int = 3) -> np.ndarray:
        rows = self._create_rows(images)
        rows = self._merge_img_in_rows(rows, channels)
        canvas = self._merge_rows_in_canvas(rows, channels)

        return canvas

    def _create_rows(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        num_images = len(images)
        rows_num, _ = self._get_grid_size(num_images)
        self._img_height = rows_num * (self._row_height + self._gap) + self._gap
        image_widths = [image.shape[1] for image in images]

        if rows_num == 1:
            one_big_row_width = sum(image_widths) + (num_images - 1) * self._gap
        else:
            one_big_row_width = sum(image_widths[:-rows_num]) + (num_images - 1) * self._gap
        self._row_width = one_big_row_width // rows_num
        if num_images == 1:
            one_big_row_width = images[0].shape[1]
            self._row_width = one_big_row_width
            return [images]

        rows = []
        row_images = []
        current_width = 0

        for idx, (image, width) in enumerate(zip(images, image_widths)):
            if current_width + width > self._row_width:
                rows.append(row_images)

                row_images = []
                current_width = 0
                if len(rows) == rows_num:
                    return rows
            row_images.append(image)
            if idx == num_images - 1:
                rows.append(row_images)
                if len(rows) == rows_num:
                    return rows
            current_width += width + self._gap

        return rows

    def _get_grid_size(self, num: int = 1, aspect_ratio: Union[float, int] = 1.9) -> tuple:
        cols = max(int(math.sqrt(num) * aspect_ratio), 1)
        rows = max((num - 1) // cols + 1, 1)
        return (rows, cols)

    def _merge_img_in_rows(self, rows: List[np.ndarray], channels: int = 3) -> List[np.ndarray]:
        max_row_width = 0
        img_widths = []
        for row in rows:
            sum_widths = sum([img.shape[1] for img in row])
            img_widths.append(sum_widths)
            min_gap = self._gap * (len(row) + 1)
            max_row_width = max(max_row_width, sum_widths + min_gap)
        self._row_width = max_row_width

        combined_rows = []
        for row, width in zip(rows, img_widths):
            if len(row) == 0:
                continue
            if len(row) == 1:
                return row
            gap = (self._row_width - width) // max((len(row) - 1), 1)
            separator = np.ones((self._row_height, gap, channels), dtype=np.uint8) * 255
            combined_images = []

            for image in row:
                combined_images.append(image)
                combined_images.append(separator)
            combined_images.pop()
            combined_image = np.hstack(combined_images)
            combined_rows.append(combined_image)

        return combined_rows

    def _merge_rows_in_canvas(self, rows, channels: int = 3) -> np.ndarray:
        canvas = np.ones([self._img_height, self._row_width, channels], dtype=np.uint8) * 255
        for i, image in enumerate(rows):
            if image.shape[1] != canvas.shape[1] - self._gap * 2:
                image = image[:, : canvas.shape[1] - self._gap * 2]

            row_start = i * (self._row_height + self._gap) + self._gap
            row_end = row_start + self._row_height
            column_start = self._gap
            column_end = image.shape[1] + self._gap

            if row_end > canvas.shape[0]:
                continue
            canvas[row_start:row_end, column_start:column_end] = image

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
        height = max(self._row_height // 5, 80)
        image2 = cv2.imread(self._logo_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGBA)
        image2 = self._resize_image(image2, height)

        height_r = height
        while image2.shape[1] > image.shape[1] * 0.25:
            height_r = int(0.95 * height_r)
            image2 = self._resize_image(image2, height_r)

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
        font_size = self._get_base_font_size(text, image_w)
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
        draw = ImageDraw.Draw(tmp_canvas)
        text_width, text_height = draw.textsize(text, font=font)
        x, y = (x_pos_center - int(text_width / 2), y_pos_percent)
        draw.text((half_offset, -half_offset // 3), text, font=font, fill=text_color)

        tmp_canvas = np.array(tmp_canvas, dtype=np.uint8)
        canvas_h, canvas_w = tmp_canvas.shape[:2]

        image[y - 3 : y + canvas_h + 3, x - 3 : x + canvas_w + 3, :3] = 255
        image[y : y + canvas_h, x : x + canvas_w, :3] = tmp_canvas

        return image

    def _get_base_font_size(self, text, image_w):
        desired_text_width = image_w * 0.85
        desired_text_height = self._row_height * 0.2
        text_height_percent = 25
        font_size = 30

        font = ImageFont.truetype(self._font, font_size)

        text_width, title_height = font.getsize(text)

        while text_width > desired_text_width or title_height > desired_text_height:
            font_size -= 1
            font = ImageFont.truetype(self._font, font_size)
            text_width, title_height = font.getsize(text)

        desired_font_height = math.ceil((self._row_height * text_height_percent) // 100)
        desired_font_size = math.ceil(font_size * desired_text_width / text_width)
        desired_font_size = min(desired_font_size, desired_font_height)

        return desired_font_size

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

    def _save_video(self, videopath: str, frames):
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
        height, width = frames[0].shape[:2]
        video_writer = cv2.VideoWriter(videopath, fourcc, 15, (width, height))

        with tqdm(desc="Saving video...", total=len(frames)) as vid_pbar:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    raise Exception("Not all frame sizes are not equal to each other.")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                video_writer.write(frame)
                vid_pbar.update(1)
        video_writer.release()

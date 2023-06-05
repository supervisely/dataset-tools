import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

import supervisely as sly
from dataset_tools.image.stats.basestats import BaseVisual
from dataset_tools.image.renders.convert import from_mp4_to_webm, compress_mp4, compress_png
from supervisely.imaging import font as sly_font

UNLABELED_COLOR = [0, 0, 0]
font_name = "FiraSans-Regular.ttf"


class ClassesPreview(BaseVisual):
    """Get previews for classes in the dataset."""

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_name: str,
        api: sly.Api = None,
        row_height: int = None,
        force: bool = False,
    ):
        self.force = force
        self._meta = project_meta
        self._project_name = project_name
        classes_cnt = len(self._meta.obj_classes)
        classes_text = "classes" if classes_cnt > 1 else "class"
        self._title = f"{self._project_name} · {classes_cnt} {classes_text}"

        self._gap = 20
        self._img_width = 1920
        self._img_height = None
        self._row_height = row_height if row_height is not None else 480
        self._row_width = None

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

    def animate(self, path: str = None):
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
                pad = "10%"
                crops = sly.aug.instance_crop(
                    img=img,
                    ann=ann,
                    class_title=cls_name,
                    save_other_classes_in_crop=False,
                    padding_config={"top": pad, "bottom": pad, "left": pad, "right": pad},
                )
                crops = sorted(crops, key=lambda crop: crop[1].labels[0].area / image_area)
                cropped_img, cropped_ann = crops[-1]

                cropped_img = self._resize_image(cropped_img, self._row_height)
                cropped_ann = cropped_ann.resize(cropped_img.shape[:2])
                ann_mask = np.zeros((*cropped_img.shape[:2], 3), dtype=np.uint8)
                text_mask = np.zeros((*cropped_img.shape[:2], 3), dtype=np.uint8)
                cropped_ann.draw_pretty(ann_mask, thickness=5, opacity=1)

                label = cropped_ann.labels[0]
                bbox = label.geometry.to_bbox()
                f_scale = self._get_optimal_font_scale(cls_name, (bbox.height, bbox.width))
                font = cv2.FONT_HERSHEY_SIMPLEX
                t_width, _ = cv2.getTextSize(cls_name, font, f_scale, thickness=3)[0][:2]

                if type(label.geometry) in [sly.Bitmap, sly.Polygon]:
                    col, row = bbox.center.col, bbox.center.row
                    org = (col - int(t_width / 2), row)
                    white = (255, 255, 255, 255)
                    cv2.putText(text_mask, cls_name, org, font, f_scale, white, 3, cv2.LINE_AA)

                else:
                    color = label.obj_class.color
                    cv2.rectangle(
                        ann_mask, (bbox.left, bbox.top), (bbox.right, bbox.bottom), color, 2
                    )
                    org = (bbox.top, bbox.left)
                    white = (255, 255, 255, 255)
                    cv2.putText(text_mask, cls_name, org, font, f_scale, white, 3, cv2.LINE_AA)
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
        rows, _ = self._get_grid_size(num_images)
        self._img_height = rows * (self._row_height + self._gap) + self._gap
        image_widths = [image.shape[1] for image in images]
        
        if rows == 1:
            one_big_row_width = sum(image_widths) + (num_images - 1) * self._gap
        else:
            one_big_row_width = sum(image_widths[:-rows]) + (num_images - 1) * self._gap
        self._row_width = one_big_row_width // rows
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
            row_images.append(image)
            if idx == num_images - 1:
                rows.append(row_images)
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

    def _get_optimal_font_scale(self, text: str, image_size: tuple) -> float:
        font_scale = 10
        thickness = 3
        text_height_percent = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        while text_width > image_size[1] * 0.6:
            font_scale -= 0.1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]

        desired_text_height = (image_size[0] * text_height_percent) // 100
        font_scale *= desired_text_height / text_height
        return font_scale

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
        image_h, image_w = image.shape[:2]
        font_scale = self._get_optimal_font_scale(text, image.shape[:2])
        font_size = sly_font.get_readable_font_size((image_h, image_w))
        font = sly_font.get_font(font_name, int(font_size * font_scale))
        l, t, r, b = font.getbbox(text)
        title_w, title_h = r - l + self._gap, b
        while title_h > self._row_height * 0.3 or title_w > image_w * 0.8:
            font = font.font_variant(size=int(font.size * 0.96))
            l, t, r, b = font.getbbox(text)
            title_w, title_h = r - l + self._gap, b
        x, y = (image_w - title_w) // 2, self._gap * 2
        left, top, right, bottom = (x, y, x + title_w, y + title_h)
        tmp_canvas = np.zeros((title_h, title_w, 3), dtype=np.uint8)
        tmp_canvas = self._gradient(tmp_canvas, 0, 0, title_w, title_h)
        color = (255, 255, 255, 255)
        sly.image.draw_text(tmp_canvas, text, (-t // 2, self._gap // 2), "tl", font, False, color)
        image[top - 3 : bottom + 3, left - 3 : right + 3, :3] = 255
        image[top:bottom, left:right, :3] = tmp_canvas
        return image

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

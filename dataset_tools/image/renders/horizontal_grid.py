import os
import random
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

import supervisely as sly
from supervisely.imaging import font as sly_font
from dataset_tools.image.renders.convert import from_mp4_to_webm, compress_mp4, compress_png


class HorizontalGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 2,
        cols: int = 6,
        force: bool = False,
        is_detection_task: bool = False,
    ):
        self.force = force
        self.project_meta = project_meta
        self._is_detection_task = is_detection_task

        self._img_height = 1920
        self._rows = rows
        self._cols = cols
        self._gap = 15
        self._row_width = 0
        self._logo_path = "logo.png"

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

                ann_mask = self._resize_image(ann_mask, self._row_height)
                img = self._resize_image(img, self._row_height)
                ann = ann.resize(img.shape[:2])
                ann: sly.Annotation
                for label in ann.labels:
                    if type(label.geometry) == sly.Point:
                        label.draw(ann_mask, thickness=15)
                        label.draw(img, thickness=15)
                    elif self._is_detection_task:
                        bbox = label.geometry.to_bbox()
                        pt1, pt2 = (bbox.left, bbox.top), (bbox.right, bbox.bottom)
                        cv2.rectangle(ann_mask, pt1, pt2, label.obj_class.color, thickness=10)
                        cv2.rectangle(img, pt1, pt2, label.obj_class.color, thickness=10)
                        font_size = int(sly_font.get_readable_font_size(img.shape[:2]) * 1.4)
                        font = sly_font.get_font(font_size=font_size)
                        _, _, _, bottom = font.getbbox(label.obj_class.name)
                        anchor = (bbox.top - bottom, bbox.left)
                        sly.image.draw_text(
                            ann_mask[:, :, :3], label.obj_class.name, anchor, font=font
                        )
                        sly.image.draw_text(img, label.obj_class.name, anchor, font=font)
                if not self._is_detection_task:
                    ann.draw_pretty(
                        ann_mask[:, :, :3], thickness=0, opacity=0.7, fill_rectangles=False
                    )
                    ann.draw_pretty(img, thickness=0, opacity=0.7, fill_rectangles=False)

                self.np_frames.append(self._resize_image(tmp, self._row_height))  # for gif
                self.np_anns.append(ann_mask)  # for gif
                # img = self._resize_image(img, self._row_height)
                self.np_images.append(img)  # for grid

    def to_image(self, path: str = None):
        path_part, ext = os.path.splitext(path)
        tmp_path = f"{path_part}-o{ext}"
        if path is None:
            storage_dir = sly.app.get_data_dir()
            path = os.path.join(storage_dir, "horizontal_grid.png")

        self._img_array = self._merge_canvas_with_images(self.np_images)
        self._add_logo(self._img_array)

        sly.image.write(tmp_path, self._img_array)
        compress_png(tmp_path, path)
        sly.fs.silent_remove(tmp_path)
        sly.logger.info(f"Result grid saved to: {path}")

    def animate(self, path: str = None):
        bg = self._merge_canvas_with_images(self.np_frames, 4)
        ann = self._merge_canvas_with_images(self.np_anns, 4)
        ann[:, :, 3] = np.where(np.all(ann == 255, axis=-1), 0, 255).astype(np.uint8)

        duration = 1.1
        fps = 15
        num_frames = int(duration * fps)
        frames = []
        for i in list(range(1, num_frames + 1)) + list(range(num_frames, 0, -1)):
            alpha = i / num_frames
            frame = self._overlay_images(bg, ann, alpha)
            self._add_logo(frame)
            frame = self._resize_image(frame, frame.shape[0] // 2)
            frames.append(frame)
            if i == num_frames:
                frames.extend([frame] * (fps // 2))

        tmp_video_path = f"{os.path.splitext(path)[0]}-o.mp4"
        video_path = f"{os.path.splitext(path)[0]}.mp4"
        self._save_video(tmp_video_path, frames)
        from_mp4_to_webm(tmp_video_path, path)
        compress_mp4(tmp_video_path, video_path)
        sly.fs.silent_remove(tmp_video_path)

        sly.logger.info(f"Animation saved to: {path}, {video_path}")

    def _merge_canvas_with_images(self, images, channels: int = 3):
        rows = self._create_rows(images)
        rows = self._merge_img_in_rows(rows, channels)

        canvas = np.ones([self._img_height, self._row_width, channels], dtype=np.uint8) * 255
        for i, image in enumerate(rows):
            if image.shape[1] > canvas.shape[1] - self._gap * 2:
                image = image[:, : canvas.shape[1] - self._gap * 2]

            row_start = i * (self._row_height + self._gap) + self._gap
            row_end = row_start + self._row_height
            column_start = self._gap
            column_end = image.shape[1] + self._gap

            if row_end > canvas.shape[0]:
                continue
            canvas[row_start:row_end, column_start:column_end] = image
        return canvas

    def _create_rows(self, images):
        num_images = len(images)
        image_widths = [image.shape[1] for image in images]

        one_big_row_width = sum(image_widths) + (num_images - 1) * self._gap
        self._row_width = one_big_row_width // self._rows

        rows = []
        row_images = []
        current_width = 0

        for idx, (image, width) in enumerate(zip(images, image_widths)):
            if current_width + width > self._row_width:
                rows.append(row_images)

                row_images = []
                current_width = 0
                if len(rows) == self._rows:
                    return rows
            row_images.append(image)
            if idx == len(images) - 1:
                rows.append(row_images)
                return rows
            current_width += width + self._gap

        return rows

    def _merge_img_in_rows(self, rows, channels: int = 3):
        max_width = 0
        img_widths = []
        for row in rows:
            sum_widths = sum([img.shape[1] for img in row])
            img_widths.append(sum_widths)
            min_gap = self._gap * (len(row) + 1)
            max_width = max(max_width, sum_widths + min_gap)
        self._row_width = max_width

        combined_rows = []

        for row, width in zip(rows, img_widths):
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

    def _resize_image(self, image, height):
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = height / img_h
        img_h = height
        img_w = int(img_aspect_ratio * img_w)

        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _add_logo(self, image):
        height = max(self._row_height // 5, 80)
        image2 = cv2.imread(self._logo_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGBA)
        image2 = self._resize_image(image2, height)

        h1, w1 = image.shape[:2]
        h2, w2 = image2.shape[:2]
        x = w1 - w2 - self._gap * 2
        y = h1 - h2 - self._gap * 2
        alpha = image2[:, :, 3] / 255.0

        reg = image[y : y + h2, x : x + w2]
        image[y : y + h2, x : x + w2, :3] = (1 - alpha[:, :, np.newaxis]) * reg[:, :, :3] + alpha[
            :, :, np.newaxis
        ] * image2[:, :, :3]

    def _overlay_images(self, bg, overlay, opacity):
        alpha = overlay[..., 3] * opacity / 255.0
        one_minus_alpha = 1 - alpha

        result_image = np.copy(bg)
        for c in range(3):
            result_image[..., c] = bg[..., c] * one_minus_alpha + overlay[..., c] * alpha
        return result_image

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

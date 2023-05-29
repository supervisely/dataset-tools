import os
import random
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

import supervisely as sly
from dataset_tools.convert.convert import from_mp4_to_webm, process_mp4, process_png


class VerticalGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 6,
        cols: int = 3,
        footer_path: str = "dninja_footer.png",
        force:bool = False,
    ):
        self.force = force
        self.project_meta = project_meta

        self._img_width = 1920
        self._rows = rows
        self._cols = cols
        self._g_spacing = 15
        self._column_height = 0
        self._footer_path = footer_path

        self.np_images = []  # for grid
        self.np_anns = []  # for gif
        self.np_frames = []  # for gif
        self._img_array = None
        self._column_width = int(
            (self._img_width - self._g_spacing * (self._cols + 1)) / self._cols
        )

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
                ann: sly.Annotation
                img = (
                    sly.image.read(ds.get_img_path(img_info.name))
                    if self._local
                    else self._api.image.download_np(img_info.id)
                )
                tmp = np.dstack((img, np.ones((*img.shape[:2], 1), dtype=np.uint8) * 255))
                ann_mask = np.ones((*img.shape[:2], 4), dtype=np.uint8) * 255
                ann.draw_pretty(ann_mask[:, :, :3], thickness=0, opacity=0.7)
                self.np_frames.append(self._resize_image(tmp, self._column_width))  # for gif
                self.np_anns.append(self._resize_image(ann_mask, self._column_width))  # for gif

                ann.draw_pretty(img, thickness=0, opacity=0.7)
                img = self._resize_image(img, self._column_width)
                self.np_images.append(img)

                p.update(1)

    def to_image(self, path: str = None):
        path_part, ext = os.path.splitext(path)
        tmp_path = f"{path_part}-o{ext}"
        if path is None:
            storage_dir = sly.app.get_data_dir()
            path = os.path.join(storage_dir, "vertical_grid.png")
        self._img_array = self._merge_canvas_with_images(self.np_images)
        self._add_footer_with_logo(self._img_array)

        sly.image.write(tmp_path, self._img_array)
        process_png(tmp_path, path, 1080)
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
            self._add_footer_with_logo(frame)
            frame = self._resize_image(frame, frame.shape[1] // 2)
            frames.append(frame)
            if i == num_frames:
                frames.extend([frame] * (fps // 2))

        tmp_video_path = f"{os.path.splitext(path)[0]}-o.mp4"
        video_path = f"{os.path.splitext(path)[0]}.mp4"
        self._save_video(tmp_video_path, frames)
        from_mp4_to_webm(tmp_video_path, path)
        process_mp4(tmp_video_path, video_path)
        sly.fs.silent_remove(tmp_video_path)

        sly.logger.info(f"Animation saved to: {path}, {video_path}")

    def _merge_canvas_with_images(self, images, channels: int = 3):
        columns = self._create_columns(images)
        columns = self._merge_img_in_columns(columns, channels)

        canvas = np.ones([self._column_height, self._img_width, channels], dtype=np.uint8) * 255
        for i, image in enumerate(columns):
            if image.shape[0] > canvas.shape[0] - self._g_spacing:
                image = image[: canvas.shape[0] - self._g_spacing, :]

            column_start = i * (self._column_width + self._g_spacing) + self._g_spacing
            column_end = column_start + self._column_width
            row_start = self._g_spacing
            row_end = canvas.shape[0]

            canvas[row_start:row_end, column_start:column_end] = image

        return canvas

    def _create_columns(self, images):
        num_images = len(images)
        image_heights = [image.shape[0] for image in images]

        one_big_column_height = (
            sum(image_heights[: -self._cols]) + (num_images - 2) * self._g_spacing
        )
        self._column_height = one_big_column_height // self._cols

        columns = []
        column_images = []
        current_height = 0

        for image, height in zip(images, image_heights):
            column_images.append(image)
            current_height += height + self._g_spacing

            if current_height > self._column_height:
                columns.append(column_images)

                column_images = []
                current_height = 0

        if len(columns) == self._cols:
            return columns
        return columns

    def _merge_img_in_columns(self, columns, channels: int = 3):
        combined_columns = []
        separator = np.ones((15, self._column_width, channels), dtype=np.uint8) * 255
        for column in columns:
            combined_images = []

            for image in column:
                combined_images.append(image)
                combined_images.append(separator)
            combined_images.pop()
            combined_image = np.vstack(combined_images)
            combined_columns.append(combined_image)

        return combined_columns

    def _resize_image(self, image, width):
        img_h, img_w = image.shape[:2]
        img_aspect_ratio = width / img_w
        img_w = int(width)
        img_h = int(img_aspect_ratio * img_h)

        image = sly.image.resize(image, (img_h, img_w))

        return image

    def _add_footer_with_logo(self, image):
        image2 = cv2.imread(self._footer_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGBA)

        height1, width1 = image.shape[:2]
        height2, width2 = image2.shape[:2]

        if width1 != width2:
            scale_factor = width1 / width2
            height2, width2 = int(scale_factor * height2), width1
            image2 = sly.image.resize(image2, (height2, width2))

        alpha_channel = image2[:, :, 3] / 255.0

        x = 0
        y = height1 - height2

        region = image[y:height1, x : x + width2]
        image[y:height1, x : x + width2, :3] = (1 - alpha_channel[:, :, np.newaxis]) * region[
            :, :, :3
        ] + alpha_channel[:, :, np.newaxis] * image2[:, :, :3]

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

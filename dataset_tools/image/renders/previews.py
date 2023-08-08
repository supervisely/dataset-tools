import os
from datetime import datetime
from PIL import Image, ImageDraw

import numpy as np

import supervisely as sly


class Previews:
    def __init__(
        self,
        project_id,
        project_meta,
        api,
        team_id,
        force: bool = False,
        is_detection_task: bool = False,
    ):
        self.project_meta = project_meta
        self.project_id = project_id
        self.force = force
        self._is_detection_task = is_detection_task

        self.MAX_WIDTH = 500
        self.BATCH_SIZE = 50

        self.api = api
        self.team_id = team_id

        self.render_dir = f"/dataset/{project_id}/renders"

        self.images_batch = []
        self.errors = []

    def update(self, image, ann):
        self.images_batch.append((image, ann))

        if len(self.images_batch) >= self.BATCH_SIZE:
            self._save_batch()
            self.images_batch.clear()

    def _save_batch(self):
        existing_renders = self.api.file.list2(self.team_id, self.render_dir, recursive=False)
        existing_ids = set(int(sly.fs.get_file_name(f.path)) for f in existing_renders)

        local_paths, remote_paths = [], []

        for image, ann in self.images_batch:
            if image.id in existing_ids and not self.force:
                continue

            if image.width > self.MAX_WIDTH:
                out_size = (int((image.height / image.width) * self.MAX_WIDTH), self.MAX_WIDTH)
                try:
                    ann = ann.resize(out_size)
                except ValueError:
                    continue

            render = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)

            if self._is_detection_task:
                rgba = self._draw_bbox(ann)

            else:
                for label in ann.labels:
                    label: sly.Label
                    if type(label.geometry) == sly.Point:
                        label.draw(render, thickness=15)
                    if type(label.geometry) != sly.Rectangle:
                        label.draw(render, thickness=ann._get_thickness())
                    else:
                        label.draw_contour(render, thickness=self._get_thickness(render))
                alpha = (1 - np.all(render == [0, 0, 0], axis=-1).astype("uint8")) * 255
                rgba = np.dstack((render, alpha))

            local_path = os.path.join(os.getcwd(), "tmp/renders", f"{image.id}.png")
            remote_path = os.path.join(self.render_dir, f"{image.id}.png")

            local_paths.append(local_path)
            remote_paths.append(remote_path)
            sly.image.write(local_path, rgba, remove_alpha_channel=False)

        try:
            self.api.file.upload_bulk(self.team_id, local_paths, remote_paths)
        except Exception as e:
            self.errors.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_ids": [image.id for image, ann in self.images_batch],
                    "error": str(e),
                }
            )

        for path in local_paths:
            sly.fs.silent_remove(path)

    def close(self):
        if len(self.images_batch) > 0:
            self._save_batch()

        if len(self.errors) > 0:
            print("Errors during renders upload:")
            for error in self.errors:
                print(error)

    def _get_thickness(self, render):
        THICKNESS_FACTOR = 0.005
        render_height, render_width, _ = render.shape
        thickness = int(max(render_height, render_width) * THICKNESS_FACTOR)
        return thickness

    def rgb_to_rgba(self, rgb: np.ndarray) -> np.ndarray:
        """Converts RGB image to RGBA image with alpha channel set to 255 for non-black pixels.

        :param rgb: RGB image as numpy array with shape (height, width, 3)
        :type rgb: np.ndarray
        :return: RGBA image as numpy array with shape (height, width, 4)
        :rtype: np.ndarray"""
        alpha = (1 - np.all(rgb == [0, 0, 0], axis=-1).astype("uint8")) * 255
        return np.dstack((rgb, alpha))

    def _draw_bbox(self, ann: sly.Annotation, fill_opacity: float = 0.3) -> np.ndarray:
        """Draws bounding boxes on transparent image with non-transparent frame and semi-transparent fill.
        Uses object class color for frame and fill.

        :param ann: Supervisely annotation
        :type ann: sly.Annotation
        :return: RGBA image as numpy array with shape (height, width, 4)
        :rtype: np.ndarray"""

        img = Image.new("RGBA", (ann.img_size[1], ann.img_size[0]), (0, 0, 0, 0))
        frame_render = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)

        for label in ann.labels:
            label: sly.Label
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            fill_color = tuple(label.obj_class.color) + (int(255 * fill_opacity),)

            bbox = label.geometry.to_bbox()
            coords = (
                (bbox.left, bbox.top),
                (bbox.right, bbox.bottom),
            )

            draw = ImageDraw.Draw(overlay)
            draw.rectangle(coords, fill=fill_color)

            ann.draw_contour(frame_render, thickness=self._get_thickness(frame_render))

            img = Image.alpha_composite(img, overlay)

        render = np.array(img)
        frame_render = self.rgb_to_rgba(frame_render)

        render = self._overlay_images(frame_render, render)

        return render

    def _overlay_images(self, top_layer: np.ndarray, bottom_layer: np.ndarray) -> np.ndarray:
        """Paste `top_layer` onto `bottom_layer` ignoring bottom layer's alpha channel.

        :param top_layer: The image that should be pasted onto the bottom layer, overlaying the bottom layer.
        :type top_layer: RGBA numpy array with shape (height, width, 4)
        :param bottom_layer: The image that should be used as the bottom layer, it will be overlaid by the top layer.
        :type bottom_layer: RGBA numpy array with shape (height, width, 4)
        :return: The resulting image.
        :rtype: RGBA numpy array with shape (height, width, 4)"""
        result = bottom_layer.copy()
        top_alpha = top_layer[:, :, 3]
        result[top_alpha == 255] = top_layer[top_alpha == 255]

        return result

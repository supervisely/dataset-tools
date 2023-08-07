import os
import cv2
from datetime import datetime
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
                        thickness = self._get_thickness(render)
                        label.draw_contour(render, thickness=thickness)
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

    def _draw_bbox(self, ann):
        FRAME_OPACITY = 1.0
        FILL_OPACITY = 0.3

        render = np.zeros((ann.img_size[0], ann.img_size[1], 4), dtype=np.uint8)

        for label in ann.labels:
            label: sly.Label
            color = tuple(label.obj_class.color)
            frame_alpha = int(255 * FRAME_OPACITY)
            fill_alpha = int(255 * FILL_OPACITY)

            bbox = label.geometry.to_bbox()
            pt1, pt2 = (bbox.left, bbox.top), (bbox.right, bbox.bottom)
            thickness = self._get_thickness(render)

            temp_render = np.zeros_like(render)

            cv2.rectangle(temp_render, pt1, pt2, (*color, frame_alpha), thickness=thickness)

            temp_render[pt1[1] : pt2[1], pt1[0] : pt2[0], 0:3] = color
            temp_render[pt1[1] : pt2[1], pt1[0] : pt2[0], 3] = fill_alpha

            render = cv2.addWeighted(render, 1.0, temp_render, 1.0, 0.0)

        return render

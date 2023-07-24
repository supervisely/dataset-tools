import os
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
                thickness_factor = 0.005
                render_height, render_width, _ = render.shape
                thickness = int(max(render_height, render_width) * thickness_factor)
                ann.draw_pretty(render, thickness=thickness, opacity=0.3)
            else:
                for label in ann.labels:
                    label: sly.Label
                    if type(label.geometry) == sly.Point:
                        label.draw(render, thickness=15)
                    elif type(label.geometry) != sly.Rectangle:
                        label.draw(render, thickness=ann._get_thickness())
                    else:
                        label.draw_contour(render, thickness=ann._get_thickness())
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


from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")
api: sly.Api = sly.Api.from_env()

project_id = os.environ.get("PROJECT_ID")
team_id = os.environ.get("TEAM_ID")
print("Project ID:", project_id)
print("Team ID:", team_id)

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

pr = Previews(project_id, project_meta, api, team_id, is_detection_task=True)

datasets = api.dataset.get_list(project_id)

images, anns = [], []

for dataset in datasets:
    dataset_images = api.image.get_list(dataset.id)
    dataset_anns_jsons = api.annotation.download_json_batch(
        dataset.id, [image.id for image in dataset_images]
    )
    dataset_anns = [
        sly.Annotation.from_json(ann_json, project_meta) for ann_json in dataset_anns_jsons
    ]

    images.extend(dataset_images)
    anns.extend(dataset_anns)

assert len(images) == len(anns)

print("Total number of images:", len(images))
print("Total number of annotations:", len(anns))
print("Type of the first annotation:", type(anns[0]))

for image, ann in zip(images, anns):
    pr.update(image, ann)

pr.close()

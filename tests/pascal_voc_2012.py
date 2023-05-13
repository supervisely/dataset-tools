import os
from typing import List

import numpy as np
from dotenv import load_dotenv

import dataset_tools as dtools
import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()

MAX_WIDTH = 500
project_id = 223

project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
workspace_id = project_info.workspace_id
workspace_info = api.workspace.get_info_by_id(workspace_id)
team_id = workspace_info.team_id

ninja_dir = f"/{workspace_id}/{project_id}/"

# render annotations
for dataset in api.dataset.get_list(project_id):
    render_dir = os.path.join(ninja_dir, "renders", f"{dataset.id}")

    existing_renders = api.file.list2(team_id, render_dir, recursive=False)
    existing_ids = set(int(sly.fs.get_file_name(f.path)) for f in existing_renders)
    all_images = api.image.get_list(dataset.id)
    new_images: List[sly.ImageInfo] = []
    for image in all_images:
        if image.id not in existing_ids:
            new_images.append(image)

    for batch in sly.batched(new_images):
        image_ids = [image.id for image in batch]
        anns = api.annotation.download_json_batch(dataset.id, image_ids)
        for jann, image in zip(anns, batch):
            out_size = (image.height, image.width)
            if image.width > MAX_WIDTH:
                out_size = (int((image.height / image.width) * MAX_WIDTH), MAX_WIDTH)

            ann = sly.Annotation.from_json(jann, project_meta)
            ann = ann.resize(out_size)
            render = np.zeros((out_size[0], out_size[1], 3), dtype=np.uint8)
            ann.draw(render, thickness=ann._get_thickness())

            local_path = os.path.join(os.getcwd(), "demo", f"{image.id}.png")
            remote_path = os.path.join(render_dir, f"{image.id}.png")
            sly.image.write(local_path, render)
            api.file.upload(team_id, local_path, remote_path)
            sly.fs.silent_remove(local_path)

import json
import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import dataset_tools as dtools
import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

MAX_WIDTH = 500

api = sly.Api.from_env()
project_id = sly.env.project_id()
workspace_id = sly.env.workspace_id()
team_id = sly.env.team_id()

project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# stats_dir = f"/dataset/{project_id}/stats"
# cls_balance = dtools.ClassBalance(project_meta)
# pbar = tqdm(total=project_info.items_count)
# for dataset in api.dataset.get_list(project_id):
#     for batch in api.image.get_list_generator(dataset.id, batch_size=100):
#         image_ids = [image.id for image in batch]
#         anns = api.annotation.download_json_batch(dataset.id, image_ids)
#         for img, jann in zip(batch, anns):
#             ann = sly.Annotation.from_json(jann, project_meta)
#             cls_balance.update(img, ann)
#             pbar.update(1)

# with open("./demo/class_balance.json", "w") as f:
#     json.dump(cls_balance.to_json(), f)

# cls_balance.to_image("./demo/class_balance.png")

# exit(0)

# coocurance -  в формат виджета +  убрать totals + clicks

# renders
render_dir = f"/dataset/{project_id}/renders"
pbar = tqdm(total=project_info.items_count)
existing_renders = api.file.list2(team_id, render_dir, recursive=False)
existing_ids = set(int(sly.fs.get_file_name(f.path)) for f in existing_renders)
pbar.update(len(existing_ids))
for dataset in api.dataset.get_list(project_id):
    ds_images = api.image.get_list(dataset.id)
    new_images: List[sly.ImageInfo] = []
    for image in ds_images:
        if image.id not in existing_ids:
            new_images.append(image)

    for batch in sly.batched(new_images):
        image_ids = [image.id for image in batch]
        anns = api.annotation.download_json_batch(dataset.id, image_ids)

        lpaths = []
        rpaths = []
        for jann, image in zip(anns, batch):
            ann = sly.Annotation.from_json(jann, project_meta)
            if image.width > MAX_WIDTH:
                out_size = (int((image.height / image.width) * MAX_WIDTH), MAX_WIDTH)
                ann = ann.resize(out_size)

            render = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
            ann.draw(render, thickness=ann._get_thickness())
            alpha = (1 - np.all(render == [0, 0, 0], axis=-1).astype("uint8")) * 255
            rgba = np.dstack((render, alpha))

            local_path = os.path.join(os.getcwd(), "demo", f"{image.id}.png")
            remote_path = os.path.join(render_dir, f"{image.id}.png")
            lpaths.append(local_path)
            rpaths.append(remote_path)
            sly.image.write(local_path, rgba, remove_alpha_channel=False)

        api.file.upload_bulk(team_id, lpaths, rpaths)
        for p in lpaths:
            sly.fs.silent_remove(p)
        pbar.update(len(lpaths))
pbar.close()

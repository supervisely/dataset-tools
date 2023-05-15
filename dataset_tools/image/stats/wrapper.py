import random
import os
from tqdm import tqdm
import json

from collections import defaultdict

import supervisely as sly


def sample_images(api, datasets, sample_rate):
    all_images = []
    for dataset in datasets:
        images = api.image.get_list(dataset.id)
        all_images.extend(images)

    cnt_images = len(all_images)
    if sample_rate != 1:
        cnt_images = int(max(1, sample_rate * len(all_images)))
        random.shuffle(all_images)
        all_images = all_images[:cnt_images]

    ds_images = defaultdict(list)
    for image_info in all_images:
        ds_images[image_info.dataset_id].append(image_info)
    return ds_images, cnt_images


def calculate(
    api, stats=None, project_id=None, project_path=None, sample_rate=1, demo_dirpath=None
) -> None:
    if project_id is not None:
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        datasets = api.dataset.get_list(project_id)

    elif project_path is not None:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)
        project_meta = project_fs.meta
        datasets = project_fs.datasets

    for stat_name, Statistics in stats.items():
        stat = Statistics(project_meta)
        dataset_sample, sample_count = sample_images(api, datasets, sample_rate)

        with tqdm(total=sample_count) as pbar:
            for dataset_id, sample in dataset_sample.items():
                for batch in sly.batched(sample, batch_size=100):
                    image_ids = [image.id for image in batch]
                    janns = api.annotation.download_json_batch(dataset_id, image_ids)

                    for img, jann in zip(batch, janns):
                        ann = sly.Annotation.from_json(jann, project_meta)
                        stat.update(img, ann)
                        pbar.update(1)

        if demo_dirpath is not None:
            os.makedirs(demo_dirpath, exist_ok=True)
            json_file_path = os.path.join(demo_dirpath, f"{stat_name}.json")
            image_file_path = os.path.join(demo_dirpath, f"{stat_name}.png")

            with open(json_file_path, "w") as f:
                json.dump(stat.to_json(), f)

            stat.to_image(image_file_path)

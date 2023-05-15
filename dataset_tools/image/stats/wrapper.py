import random
import os
from tqdm import tqdm
import json

from collections import defaultdict

import supervisely as sly


def sample_images(api, datasets, sample_rate):
    all_images = []
    for dataset in datasets:
        try:
            images = api.image.get_list(dataset.id)
        except AttributeError:
            images = [
                dataset.get_image_info(sly.fs.get_file_name(img))
                for img in os.listdir(dataset.ann_dir)
            ]
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


def initialize(project_id=None, project_path=None):
    api = sly.Api.from_env()

    if project_id is not None:
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        datasets = api.dataset.get_list(project_id)

    elif project_path is not None:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)
        project_meta = project_fs.meta
        datasets = project_fs.datasets

    return project_meta, datasets


def get_stats(stats, project_meta, datasets, sample_rate=1) -> None:
    api = sly.Api.from_env()

    for Statistics in stats:
        dataset_sample, sample_count = sample_images(api, datasets, sample_rate)

        with tqdm(total=sample_count) as pbar:
            for dataset_id, sample in dataset_sample.items():
                for batch in sly.batched(sample, batch_size=100):
                    image_ids = [image.id for image in batch]
                    janns = api.annotation.download_json_batch(dataset_id, image_ids)

                    for img, jann in zip(batch, janns):
                        ann = sly.Annotation.from_json(jann, project_meta)
                        Statistics.update(img, ann)
                        pbar.update(1)

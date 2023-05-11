import random
import os

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


def get_sample(images, sample_rate):
    cnt = int(max(1, sample_rate * len(images)))
    random.shuffle(images)
    images = images[:cnt]
    return images


def calculate(api, cfg=None, project_id=None, project_dir=None, sample_rate=0.1):
    result = {}

    if project_id is not None:
        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        for statsType, Statistics in cfg.items():
            stats = {}
            datasets = api.dataset.get_list(project_id)
            ds_images, sample_count = sample_images(api, datasets, sample_rate)

            Statistics.prepare_data(stats, meta)

            for dataset_id, images in ds_images.items():
                dataset = api.dataset.get_info_by_id(dataset_id)

                for img_batch in sly.batched(images):
                    image_ids = [image_info.id for image_info in img_batch]
                    ann_batch = api.annotation.download_batch(dataset.id, image_ids)

                    for image_info, ann_info in zip(img_batch, ann_batch):
                        #  maybe *args **kwargs?
                        Statistics.update(stats, image_info, ann_info, meta, dataset)

            Statistics.aggregate_calculations(stats)

            result[statsType] = stats

    elif project_dir is not None:
        project_fs = sly.read_single_project(project_dir)
        meta = project_fs.meta

        for statsType, Statistics in cfg.items():
            stats = {}
            datasets = project_fs.datasets

            for dataset in datasets:
                images = [
                    dataset.get_image_info(sly.fs.get_file_name(img))
                    for img in os.listdir(dataset.ann_dir)
                ]
                images = get_sample(images, sample_rate) if sample_rate is not None else images

                Statistics.prepare_data(stats, meta)

                for img_batch in sly.batched(images):
                    image_ids = [image_info.id for image_info in img_batch]

                    ann_batch = api.annotation.download_batch(img_batch[0].dataset_id, image_ids)

                    for image_info, ann_info in zip(img_batch, ann_batch):
                        #  maybe *args **kwargs?
                        Statistics.update(stats, image_info, ann_info, meta, dataset)

            Statistics.aggregate_calculations(stats)

            result[statsType] = stats

    return result

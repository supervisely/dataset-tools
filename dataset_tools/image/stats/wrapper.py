import random
import os
from tqdm import tqdm
import json

from collections import defaultdict

import supervisely as sly


def read_project_data(project_id: int = None, project_dir: str = None) -> ProjectData:
    if project_dir:
        project_fs = sly.Project(project_dir, sly.OpenMode.READ)

        project_meta = project_fs.meta

        datasets = project_fs.datasets
        datasets_data = []

        for dataset in datasets:
            ann_dir = dataset.ann_dir
            image_infos_dir = dataset.item_info_dir

            ann_paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir)]
            anns_json = [json.load(open(ann_path)) for ann_path in ann_paths]

            image_infos_paths = [
                os.path.join(image_infos_dir, f) for f in os.listdir(image_infos_dir)
            ]

            image_infos_json = [
                json.load(open(image_info_path)) for image_info_path in image_infos_paths
            ]

            anns = [sly.Annotation.from_json(ann_json, project_meta) for ann_json in anns_json]
            image_infos = [sly.ImageInfo(**image_info_json) for image_info_json in image_infos_json]

            dataset_data = DatasetData(name=dataset.name, image_infos=image_infos, anns=anns)
            datasets_data.append(dataset_data)

        project_data = ProjectData(project_meta=project_meta, datasets=datasets_data)

    elif project_id:
        project_meta_json = api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta_json)

        datasets = api.dataset.get_list(project_id)
        datasets_data = []

        for dataset in datasets:
            image_infos = api.image.get_list(dataset.id)

            ann_jsons = api.annotation.download_json_batch(
                dataset.id, [image_info.id for image_info in image_infos]
            )

            anns = [sly.Annotation.from_json(ann_json, project_meta) for ann_json in ann_jsons]

            dataset_data = DatasetData(name=dataset.name, image_infos=image_infos, anns=anns)
            datasets_data.append(dataset_data)

        project_data = ProjectData(project_meta=project_meta, datasets=datasets_data)

    return project_data


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
    stats=None, project_id=None, project_dir=None, sample_rate=1, demo_dirpath=None
) -> None:
    if project_id is not None:
        api = sly.Api.from_env()
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

        for stat_name, Statistics in stats.items():
            stat = Statistics(project_meta)

            api.image.get_list_all_pages_generator

            datasets = api.dataset.get_list(project_id)
            dataset_sample, sample_count = sample_images(api, datasets, sample_rate)

            pbar = tqdm(total=sample_count)
            for dataset_id, sample in dataset_sample.items():
                for batch in api.image.get_list_generator(dataset_id, batch_size=100):
                    image_ids = [image.id for image in sample]
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

    elif project_dir is not None:
        pass

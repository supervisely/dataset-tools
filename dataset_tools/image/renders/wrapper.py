import os
import random
from typing import List, Union

import supervisely as sly


def sample_images(
    api: sly.Api,
    project: Union[int, str],
    datasets: List[Union[sly.DatasetInfo, sly.Project.DatasetDict]],
    meta: sly.ProjectMeta,
    sample_cnt: float,
):
    total = 0
    classification_only = False
    samples = []
    datasets_with_labels = []

    for dataset in datasets:
        ds_images = (
            api.image.get_list(
                dataset.id,
                filters=[{"field": "labelsCount", "operator": ">", "value": 0}],
            )
            if isinstance(project, int)
            else [
                dataset.get_image_info(sly.fs.get_file_name(img))
                for img in os.listdir(dataset.ann_dir)
                if dataset.get_image_info(sly.fs.get_file_name(img)).labels_count > 0
            ]
        )
        if len(ds_images) < sample_cnt // len(datasets):
            continue
        datasets_with_labels.append((dataset, ds_images))

    if len(datasets_with_labels) == 0:
        datasets_without_labels = []
        for dataset in datasets:
            ds_images = (
                api.image.get_list(dataset.id)
                if isinstance(project, int)
                else [
                    dataset.get_image_info(sly.fs.get_file_name(img))
                    for img in os.listdir(dataset.ann_dir)
                ]
            )
            datasets_without_labels.append((dataset, ds_images))

        datasets_with_labels = datasets_without_labels  # classification-only case
        classification_only = True

    for dataset, ds_images in datasets_with_labels:
        dataset: sly.Dataset
        k = min(
            sample_cnt // len(datasets_with_labels),
            (dataset.items_count if isinstance(project, int) else len(os.listdir(dataset.ann_dir))),
        )

        s = random.sample(ds_images, min(k, len(ds_images)))
        anns = (
            [
                sly.Annotation.from_json(ann_json, meta)
                for ann_json in api.annotation.download_json_batch(
                    dataset.id,
                    [item.id for item in s],
                )
            ]
            if isinstance(project, int)
            else [dataset.get_ann(img.name, meta) for img in s]
        )
        samples.append((dataset, s, anns))
        total += len(s)

    total = -1 if classification_only else total
    return samples, total


def prepare_renders(
    project: Union[int, str], renderers: list, sample_cnt: int = 25, api: sly.Api = None
) -> None:
    if len(renderers) == 0:
        print(
            "Passed 'renderers' parameter is empty. Enable 'force' flag to overwrite renderers output file. Skipping renderers preparation..."
        )
        return
    if api is None:
        api = sly.Api.from_env()
    if isinstance(project, int):
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project))
        datasets = api.dataset.get_list(project)
    elif isinstance(project, str):
        project_fs = sly.Project(project, sly.OpenMode.READ)
        project_meta = project_fs.meta
        datasets = project_fs.datasets
    else:
        raise ValueError("Project should be either an integer project ID or a string project path.")

    samples, total = sample_images(api, project, datasets, project_meta, sample_cnt)
    if len(samples) == 0:
        raise Exception("There are not any images with labels on them in the project.")

    if total == -1:
        poster = renderers[0]
        poster.update_unlabeled(samples)
    else:
        for renderer in renderers:
            renderer.update(samples)

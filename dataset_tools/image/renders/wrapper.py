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
    samples = []
    for dataset in datasets:
        dataset: sly.Dataset
        k = min(
            sample_cnt // len(datasets),
            (dataset.items_count if isinstance(project, int) else len(os.listdir(dataset.ann_dir))),
        )

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
        total += k
    return samples, total


def prepare_renders(
    project: Union[int, str], renderers: list, sample_cnt: int = 25, api: sly.Api = None
) -> None:
    if len(renderers) == 0:
        print("Passed 'renderers' parameter is empty. Enable 'force' flag to overwrite renderers output file. Skipping renderers preparation...")
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

    for renderer in renderers:
        renderer.update(samples)

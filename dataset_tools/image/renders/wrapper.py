import random
from typing import Union

from tqdm import tqdm

import supervisely as sly


def prepare_renders(project: Union[int, str], renderers: list, api: sly.Api = None) -> None:
    if isinstance(project, int):
        if api is None:
            api = sly.Api.from_env()

        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project))

        limit = 30 # take <limit> images from each dataset in project for rendering
        samples = []
        for dataset in api.dataset.get_list(project):
            ds_images = api.image.get_list(
                dataset.id,
                filters=[{"field": "labelsCount", "operator": ">", "value": "0"}],
            )
            if len(ds_images) == 0:
                continue
            sample_image_infos = random.sample(ds_images, limit)
            ds_anns_infos = api.annotation.download_json_batch(
                dataset.id, [s.id for s in sample_image_infos]
            )
            ds_anns = [sly.Annotation.from_json(a, project_meta) for a in ds_anns_infos]
            samples.append((dataset.id, sample_image_infos, ds_anns))

        if len(samples) == 0:
            raise Exception("There are not any images with labels on them in the project.")

        for renderer in renderers:
            renderer.update(samples)

    elif isinstance(project, str):
        raise NotImplementedError()
    else:
        raise ValueError("TODO explain here")

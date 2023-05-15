import json
import os
import random
from collections import defaultdict
from typing import Union

from tqdm import tqdm

import supervisely as sly


def count_stats(
    projeсt: Union[int, str], stats: list, sample_rate: float = 1, api: sly.Api = None
) -> None:
    if sample_rate < 0 or sample_rate > 1:
        raise ValueError("Sample rate has to be in range [0, 1]")

    if isinstance(projeсt, int):
        if api is None:
            api = sly.Api.from_env()

        project_info = api.project.get_info_by_id(projeсt, raise_error=True)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(projeсt))

        with tqdm(desc="Calculating stats", total=project_info.items_count) as pbar:
            for dataset in api.dataset.get_list(projeсt):
                ds_images = api.image.get_list(dataset.id)
                k = int(max(1, sample_rate * len(ds_images)))
                images = random.sample(ds_images, k)
                for batch in sly.batched(images):
                    image_ids = [image.id for image in batch]
                    janns = api.annotation.download_json_batch(dataset.id, image_ids)
                    for img, jann in zip(batch, janns):
                        ann = sly.Annotation.from_json(jann, project_meta)
                        for stat in stats:
                            stat.update(img, ann)
                        pbar.update(1)

    elif isinstance(projeсt, str):
        raise NotImplementedError()
    else:
        raise ValueError("TODO explain here")

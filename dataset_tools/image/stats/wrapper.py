import os
import random
from typing import Union, List

from tqdm import tqdm

import supervisely as sly


def sample_images(
    api: sly.Api,
    project: Union[int, str],
    datasets: List[Union[sly.DatasetInfo, sly.Project.DatasetDict]],
    sample_rate: float,
):
    total = 0
    samples = []
    for dataset in datasets:
        k = int(
            max(
                1,
                sample_rate
                * (
                    dataset.items_count
                    if isinstance(project, int)
                    else len(os.listdir(dataset.ann_dir))
                ),
            )
        )

        ds_images = (
            api.image.get_list(dataset.id)
            if isinstance(project, int)
            else [
                dataset.get_image_info(sly.fs.get_file_name(img))
                for img in os.listdir(dataset.ann_dir)
            ]
        )

        s = random.sample(ds_images, k)
        samples.append((dataset, s))
        total += k
    return samples, total


def count_stats(
    project: Union[int, str], stats: list, sample_rate: float = 1, api: sly.Api = None
) -> None:
    """
    Count dtools statistics instances passed as a list.

    :param project: Supervisely project ID or a local project path.
    :type project: Union[int, str]
    :param stats: list of instances of statistics
    :type stats: list
    :param sample_rate: Modify size of a statistics sample.
    :type sample_rate: float, optional
    :param api: Supervisely API
    :type api: sly.Api, optional

    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
           load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        api = sly.Api.from_env()

        project_id = sly.env.project_id()
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        datasets = api.dataset.get_list(project_id)

        stats = [
            dtools.ClassesPerImage(project_meta, datasets),
            dtools.ClassBalance(project_meta),
            dtools.ClassCooccurrence(project_meta),
        ]
        dtools.count_stats(
            project_id,
            stats=stats,
            sample_rate=0.01,
        )
        print("Saving stats...")
        for stat in stats:
            with open(f"./stats/{stat.basename_stem}.json", "w") as f:
                json.dump(stat.to_json(), f)
            stat.to_image(f"./stats/{stat.basename_stem}.png")
        print("Done")
    """
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("Sample rate has to be in range (0, 1]")
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

    samples, total = sample_images(api, project, datasets, sample_rate)
    desc = "Calculating stats" + (f" [sample={sample_rate}]" if sample_rate != 1 else "")
    with tqdm(desc=desc, total=total) as pbar:
        for dataset, images in samples:
            for batch in sly.batched(images):
                image_ids = [image.id for image in batch]

                janns = api.annotation.download_json_batch(
                    (dataset.id if isinstance(project, int) else batch[0].dataset_id), image_ids
                )

                for img, jann in zip(batch, janns):
                    ann = sly.Annotation.from_json(jann, project_meta)
                    for stat in stats:
                        stat.update(img, ann)
                    pbar.update(1)

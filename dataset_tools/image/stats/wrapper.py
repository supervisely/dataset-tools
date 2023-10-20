import gc
import multiprocessing
import os
import random
from typing import List, Union

import supervisely as sly
import tqdm
from memory_profiler import profile

from dataset_tools import (
    ClassBalance,
    ClassCooccurrence,
    ClassesPerImage,
    ObjectsDistribution,
)

CLASSES_TO_OPTIMIZE = [ClassBalance, ClassCooccurrence, ClassesPerImage, ObjectsDistribution]
MAX_HEIGHT = 500
MAX_WIDTH = 500

NUM_PROCESSING = multiprocessing.cpu_count()


def sample_images(
    api: sly.Api,
    project: Union[int, str],
    project_stats: dict,
    datasets: List[Union[sly.DatasetInfo, sly.Project.DatasetDict]],
    sample_rate: float,
):
    total = 0
    samples = []
    image_stats, imageTag_stats, objectTag_stats = project_stats['images']['datasets'], project_stats['imageTags']['datasets'], project_stats['objectTags']['datasets']
    
    for dataset, image_stat, imageTag_stat, objectTag_stat  in zip(datasets, image_stats, imageTag_stats, objectTag_stats):
        is_unlabeled = image_stat['imagesMarked'] == 0 and imageTag_stat['imagesTagged'] == 0 and objectTag_stat['objectsTagged'] == 0
        if dataset.items_count == 0 or is_unlabeled:
            continue
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
    project: Union[int, str], project_stats: dict, stats: list, sample_rate: float = 1, api: sly.Api = None
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
    if len(stats) == 0:
        print(
            "Passed 'stats' parameter is empty. Enable 'force' flag to overwrite statistics output file. Skipping statistics counting..."
        )
        return
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

    samples, total = sample_images(api, project, project_stats, datasets, sample_rate)
    desc = "Calculating stats" + (f" [sample={sample_rate}]" if sample_rate != 1 else "")
    # sly.logger.info(f"CPU count: {NUM_PROCESSING}")
    with tqdm.tqdm(desc=desc, total=total) as pbar:
        
        for dataset, images in samples:
            for batch in sly.batched(images, 100):
                image_ids = [image.id for image in batch]
                image_names = [image.name for image in batch]

                if isinstance(project, int):
                    janns = api.annotation.download_json_batch(dataset.id, [id for id in image_ids])
                    anns = [sly.Annotation.from_json(ann_json, project_meta) for ann_json in janns]
                else:
                    anns = [dataset.get_ann(name, project_meta) for name in image_names]

                # resized_anns = [resize_ann_with_aspect_ratio(ann) for ann in anns]
                # FIXME: optimization is broken (resize labels area 0 px)

                # TODO multiprocessing
                # if isinstance(stat, ClassBalance):
                #     stat.parallel_update(batch, anns, NUM_PROCESSING)
                #     pbar.update(len(batch))

                for img, ann in zip(batch, anns):
                    # pbar.set_postfix_str(img.name) #? for debug
                    for stat in stats:
                        # if stat.__class__ in CLASSES_TO_OPTIMIZE:
                        #     stat.update(img, resized_anns)
                        # else:
                        stat.update(img, ann)
                    pbar.update(1)



def resize_ann_with_aspect_ratio(ann: sly.Annotation):
    height, width = ann.img_size
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        aspect_ratio = width / height
        target_aspect_ratio = MAX_WIDTH / MAX_HEIGHT
        if aspect_ratio > target_aspect_ratio:
            new_width = MAX_WIDTH
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = MAX_HEIGHT
            new_width = int(new_height * aspect_ratio)
        new_ann = ann.resize((new_width, new_height))
        return new_ann
    return ann

from __future__ import annotations

import random
from typing import Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import supervisely as sly
from supervisely import OpenMode, Project
from supervisely._utils import batched, is_development
from supervisely.annotation.annotation import Annotation, TagCollection
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress


def get_sample_image_infos(
    api,
    project_info: sly.ProjectInfo,
    project_stats: dict,
    class_balance_json: dict = None,
    is_classification_cvtask=False,
) -> List[sly.ImageInfo]:
    MAX_WEIGHT_BYTES = 5e8
    MAX_ITEMS_COUNT = 1e3

    mean_size = int(project_info.size) / project_info.items_count
    MIN_ITEMS_COUNT_PER_CLASS = 5 if mean_size > 1e6 else 10
    total_num_classes = len(project_stats["images"]["objectClasses"])
    MIN_ITEMS_COUNT_PER_CLASS = (
        1 if mean_size > 1e6 and total_num_classes > 40 else MIN_ITEMS_COUNT_PER_CLASS
    )

    if (
        int(project_info.size) < MAX_WEIGHT_BYTES
        and int(project_info.items_count) < MAX_ITEMS_COUNT
    ) or int(project_info.size) < 7e8:
        return None

    datasets = api.dataset.get_list(project_info.id)

    optimal_size = min(MAX_ITEMS_COUNT * mean_size, MAX_WEIGHT_BYTES)
    optimal_items_count = int(optimal_size / mean_size)

    if class_balance_json is None or total_num_classes > 1000 or is_classification_cvtask:
        full_list = []
        for dataset in datasets:
            full_list += api.image.get_list(dataset.id)
        return_count = (
            optimal_items_count if len(full_list) > optimal_items_count else len(full_list)
        )
        return random.sample(full_list, return_count)

    classes_on_marked_images_sum = sum(row[1] for row in class_balance_json["data"])
    images_marked_sum = project_stats["images"]["total"]["imagesMarked"]
    images_not_marked_sum = project_stats["images"]["total"]["imagesNotMarked"]

    value_factor = (
        images_marked_sum / images_not_marked_sum
        if images_not_marked_sum > images_marked_sum
        else 1
    )

    classes_per_image = classes_on_marked_images_sum / images_marked_sum

    classes_on_all_images_sum = (
        classes_on_marked_images_sum + images_not_marked_sum * classes_per_image
    )
    sample_factor = optimal_items_count / classes_on_all_images_sum

    demo_data = []
    for class_row, img_references in zip(
        class_balance_json["data"],
        class_balance_json["referencesRow"],
    ):
        class_items_count = class_row[1]
        sampled_items_count = max(MIN_ITEMS_COUNT_PER_CLASS, int(class_items_count * sample_factor))

        if len(img_references) > sampled_items_count:
            indices = sorted(random.sample(range(len(img_references)), sampled_items_count))
            demo_data += [img_references[i] for i in indices]
        else:
            demo_data += img_references

    shrinkage_factor = len(set(demo_data)) / len(demo_data)
    demo_data = list(set(demo_data))

    imagesNotMarked_by_ds = {
        ds["id"]: int(
            ds["imagesNotMarked"]
            * sample_factor
            * classes_per_image
            * shrinkage_factor
            * value_factor
        )
        for ds in project_stats["images"]["datasets"]
    }

    img_infos_sample = []
    for dataset in datasets:
        full_list = api.image.get_list(dataset.id)

        marked = [img for img in full_list if img.id in demo_data]

        notmarked = [img_info for img_info in full_list if img_info.labels_count == 0]
        notmarked = random.sample(notmarked, imagesNotMarked_by_ds[dataset.id])

        img_infos_sample += marked + notmarked

    return img_infos_sample


def download_sample_image_project(
    api,
    project_id,
    image_infos: List[sly.ImageInfo],
    dest_dir,
    dataset_ids=None,
    log_progress=False,
    batch_size=10,
    only_image_tags=False,
    save_image_info=False,
    save_images=True,
    progress_cb=None,
):
    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = Project(dest_dir, OpenMode.CREATE)
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    project_fs.set_meta(meta)

    if only_image_tags is True:
        id_to_tagmeta = meta.tag_metas.get_id_mapping()

    for dataset_info in api.dataset.get_list(project_id):
        dataset_id = dataset_info.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        dataset_fs = project_fs.create_dataset(dataset_info.name)

        images = [image for image in image_infos if image.dataset_id == dataset_id]

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset_info.name),
                total_cnt=len(images),
            )

        for batch in batched(images, batch_size):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download images in numpy format
            if save_images:
                batch_imgs_bytes = api.image.download_bytes(dataset_id, image_ids)
            else:
                batch_imgs_bytes = [None] * len(image_ids)

            # download annotations in json format
            if only_image_tags is False:
                ann_infos = api.annotation.download_batch(dataset_id, image_ids)
                ann_jsons = [ann_info.annotation for ann_info in ann_infos]
            else:
                ann_jsons = []
                for image_info in batch:
                    tags = TagCollection.from_api_response(
                        image_info.tags, meta.tag_metas, id_to_tagmeta
                    )
                    tmp_ann = Annotation(
                        img_size=(image_info.height, image_info.width), img_tags=tags
                    )
                    ann_jsons.append(tmp_ann.to_json())

            for img_info, name, img_bytes, ann in zip(
                batch, image_names, batch_imgs_bytes, ann_jsons
            ):
                dataset_fs.add_item_raw_bytes(
                    item_name=name,
                    item_raw_bytes=img_bytes if save_images is True else None,
                    ann=ann,
                    img_info=img_info if save_image_info is True else None,
                )

            if log_progress:
                ds_progress.iters_done_report(len(batch))
            if progress_cb is not None:
                progress_cb(len(batch))

from __future__ import annotations

import os
import random
import shutil
from collections import namedtuple
from enum import Enum
from typing import Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import supervisely as sly
from supervisely import OpenMode, Project
from supervisely._utils import abs_url, batched, is_development
from supervisely.annotation.annotation import ANN_EXT, Annotation, TagCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.collection.key_indexed_collection import (
    KeyIndexedCollection,
    KeyObject,
)
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.io.fs import (
    copy_file,
    dir_empty,
    dir_exists,
    ensure_base_path,
    file_exists,
    get_file_name_with_ext,
    get_subdirs,
    list_dir_recursively,
    list_files,
    list_files_recursively,
    mkdir,
    silent_remove,
)
from supervisely.io.fs_cache import FileCache
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


def get_sample_image_infos(api, project_info, project_stats, class_balance_json):
    datasets = api.dataset.get_list(project_info.id)
    mean_size = int(project_info.size) / project_info.items_count

    MAX_WEIGHT_BYTES = 5e8
    MAX_ITEMS_COUNT = 1e3
    MIN_ITEMS_COUNT_PER_CLASS = 10

    optimal_size = min(MAX_ITEMS_COUNT * mean_size, MAX_WEIGHT_BYTES)
    optimal_items_count = int(optimal_size / mean_size)

    classes_on_images_sum = sum(row[1] for row in class_balance_json["data"])
    images_marked_sum = project_stats["images"]["total"]["imagesMarked"]
    images_not_marked_sum = project_stats["images"]["total"]["imagesNotMarked"]

    classes_per_image = classes_on_images_sum / images_marked_sum

    num_classes_on_images_sum = classes_on_images_sum + images_not_marked_sum * classes_per_image
    sample_factor = optimal_items_count / num_classes_on_images_sum

    demo_data = []
    for class_row, img_references, ds_references in zip(
        class_balance_json["data"],
        class_balance_json["referencesRow"],
        class_balance_json["referencesRowDataset"],
    ):
        class_items_count = class_row[1]
        sampled_items_count = max(MIN_ITEMS_COUNT_PER_CLASS, int(class_items_count * sample_factor))
        pairs = []
        for index, img_id in enumerate(img_references):
            nearest_lower_index = max(filter(lambda x: int(x) <= index, ds_references.keys()))
            ds_id = ds_references[nearest_lower_index]
            pairs.append((ds_id, img_id))

        if len(pairs) > sampled_items_count:
            indices = sorted(random.sample(range(len(pairs)), sampled_items_count))
            demo_data += [pairs[i] for i in indices]
        else:
            demo_data += pairs

    imagesMarked_by_ds = {
        x: list(set([img_id for ds_id, img_id in demo_data if ds_id == x]))
        for x in [dataset.id for dataset in datasets]
    }
    after_shrinkage = {
        k: len(v) / (images_marked_sum * sample_factor * classes_per_image)
        for k, v in imagesMarked_by_ds.items()
    }
    shrinkage_factor = sum([v for v in after_shrinkage.values()])

    imagesNotMarked_by_ds = {
        ds["id"]: int(ds["imagesNotMarked"] * sample_factor * classes_per_image * shrinkage_factor)
        for ds in project_stats["images"]["datasets"]
    }

    img_infos_sample = []
    for dataset in datasets:
        full_list = api.image.get_list(dataset.id)
        notmarked = [img_info for img_info in full_list if img_info.labels_count == 0]
        notmarked = random.sample(notmarked, imagesNotMarked_by_ds[dataset.id])

        img_infos_sample += [
            image for image in full_list if image.id in imagesMarked_by_ds[dataset.id]
        ] + notmarked

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
    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    if only_image_tags is True:
        id_to_tagmeta = meta.tag_metas.get_id_mapping()

    for dataset_info in api.dataset.get_list(project_id):
        dataset_id = dataset_info.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        dataset_fs = project_fs.create_dataset(dataset_info.name)
        # images = api.image.get_list(dataset_id)

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

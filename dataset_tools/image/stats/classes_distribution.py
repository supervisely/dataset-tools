import os
from dotenv import load_dotenv
import numpy as np
import random

import itertools
from collections import defaultdict

import supervisely as sly

# import dtoolz.image.stats.sly_globals as g


# import dtoolz as dtz


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()

PROJECT_ID = sly.env.project_id()
BG_COLOR = [0, 0, 0]
SAMPLE_PERCENT = 0.1

# stats = dtz.statistics.classes_distribution(project_id=777)
# stats = dtz.statistics.classes_distribution(project_path="/home/max/lemons")
# # to team files
# # or
# dtz.statistics.plot(stats)


def sample_images(api, datasets):
    all_images = []
    for dataset in datasets:
        images = api.image.get_list(dataset.id)
        all_images.extend(images)

    cnt_images = len(all_images)
    if SAMPLE_PERCENT != 100:
        cnt_images = int(max(1, SAMPLE_PERCENT * len(all_images) / 100))
        random.shuffle(all_images)
        all_images = all_images[:cnt_images]

    ds_images = defaultdict(list)
    for image_info in all_images:
        ds_images[image_info.dataset_id].append(image_info)
    return ds_images, cnt_images


def get_overviewTable(tb: dict, image_info, ann_info, meta):
    ann_json = ann_info.annotation
    ann_objects = [(obj["id"], obj["classTitle"]) for obj in ann_json["objects"]]

    ann = sly.Annotation.from_json(ann_json, meta)

    render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
    render_idx_rgb[:] = BG_COLOR

    ann.draw_class_idx_rgb(render_idx_rgb, tb["_name_to_index"])

    stat_area = sly.Annotation.stat_area(
        render_idx_rgb, tb["class_names"], tb["class_indices_colors"]
    )
    stat_count = ann.stat_class_count(tb["class_names"])

    if stat_area["unlabeled"] > 0:
        stat_count["unlabeled"] = 1

    for idx, class_name in enumerate(tb["class_names"]):
        cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
        cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0

        tb["sum_class_area_per_image"][idx] += cur_area
        tb["sum_class_count_per_image"][idx] += cur_count
        tb["count_images_with_class"][idx] += 1 if stat_count[class_name] > 0 else 0

        if class_name == "unlabeled":
            continue
        if stat_count[class_name] > 0:
            tb["image_counts_filter_by_id"][idx].append(image_info.id)
        if stat_count[class_name] > 0:
            obj_ids = [obj[0] for obj in ann_objects if obj[1] == class_name]
            tb["object_counts_filter_by_id"][idx].extend(obj_ids)

    ann = sly.Annotation.from_json(ann_json, meta)

    tb["images_count"] = tb["count_images_with_class"]
    tb["objects_count"] = tb["sum_class_count_per_image"]

    return tb


def get_cooccurenceTable(tb: dict, image_info, ann_info, meta):
    ann_json = ann_info.annotation

    ann = sly.Annotation.from_json(ann_json, meta)

    classes_on_image = set()
    for label in ann.labels:
        classes_on_image.add(label.obj_class.name)

    all_pairs = set(
        frozenset(pair) for pair in itertools.product(classes_on_image, classes_on_image)
    )
    for p in all_pairs:
        tb["counters"][p].append((image_info, tb["dataset"]))

    return tb


def update_classes_distribution(stats: dict, image_info, ann_info):
    overviewTable = get_overviewTable(stats["overviewTable"], image_info, ann_info, stats["meta"])
    stats["overviewTable"].update(overviewTable)

    cooccurenceTable = get_cooccurenceTable(
        stats["cooccurenceTable"], image_info, ann_info, stats["meta"]
    )
    stats["cooccurenceTable"].update(cooccurenceTable)


def aggregate_calculations(stats):
    # overviewTable
    with np.errstate(divide="ignore"):
        avg_nonzero_area = np.divide(
            stats["overviewTable"]["sum_class_area_per_image"],
            stats["overviewTable"]["count_images_with_class"],
        )
        avg_nonzero_count = np.divide(
            stats["overviewTable"]["sum_class_count_per_image"],
            stats["overviewTable"]["count_images_with_class"],
        )

    avg_nonzero_area = np.where(np.isnan(avg_nonzero_area), None, avg_nonzero_area)
    avg_nonzero_count = np.where(np.isnan(avg_nonzero_count), None, avg_nonzero_count)

    stats["overviewTable"]["avg_nonzero_area"] = avg_nonzero_area
    stats["overviewTable"]["avg_nonzero_count"] = avg_nonzero_count

    # cooccurenceTable
    pd_data = []
    class_names = stats["cooccurenceTable"]["class_names"]
    columns = ["name", *class_names]
    for cls_name1 in class_names:
        cur_row = [cls_name1]
        for cls_name2 in class_names:
            key = frozenset([cls_name1, cls_name2])
            imgs_cnt = len(stats["cooccurenceTable"]["counters"][key])
            cur_row.append(imgs_cnt)
        pd_data.append(cur_row)

    pd_data[:0] = [columns]
    stats["cooccurenceTable"]["pd_data"] = pd_data


def classes_distribution(project_id: int = None, project_path: str = None):
    # explicit structure
    stats = {
        "overviewTable": {
            "sample": SAMPLE_PERCENT,
            "class_names": [],
            "images_count": [],
            "image_counts_filter_by_id": [],
            "objects_count": [],
            "object_counts_filter_by_id": [],
            "avg_nonzero_area": [],
            "avg_nonzero_count": [],
        },
        "cooccurenceTable": {
            "class_names": [],
            "counters": [],
            "pd_data": [],
        },
    }

    if project_id is not None:
        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)
        stats["meta"] = meta

        # overviewTable
        class_names = ["unlabeled"]
        class_colors = [[0, 0, 0]]
        class_indices_colors = [[0, 0, 0]]
        _name_to_index = {}
        for idx, obj_class in enumerate(meta.obj_classes):
            class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            class_index = idx + 1
            class_indices_colors.append([class_index, class_index, class_index])
            _name_to_index[obj_class.name] = class_index

        stats["overviewTable"]["class_names"] = class_names
        stats["overviewTable"]["class_indices_colors"] = class_indices_colors
        stats["overviewTable"]["_name_to_index"] = _name_to_index

        stats["overviewTable"]["sum_class_area_per_image"] = [0] * len(class_names)
        stats["overviewTable"]["sum_class_count_per_image"] = [0] * len(class_names)
        stats["overviewTable"]["count_images_with_class"] = [0] * len(class_names)

        stats["overviewTable"]["image_counts_filter_by_id"] = [[] for _ in class_names]
        stats["overviewTable"]["object_counts_filter_by_id"] = [[] for _ in class_names]

        # cooccurenceTable
        class_names = [cls.name for cls in meta.obj_classes]
        counters = defaultdict(list)
        stats["cooccurenceTable"]["class_names"] = class_names
        stats["cooccurenceTable"]["counters"] = counters

        datasets = api.dataset.get_list(PROJECT_ID)
        ds_images, sample_count = sample_images(api, datasets)

        for dataset in api.dataset.get_list(project_id):
            stats["cooccurenceTable"]["dataset"] = dataset
            images = api.image.get_list(dataset.id)

            for img_batch in sly.batched(images):
                image_ids = [image_info.id for image_info in img_batch]
                ann_batch = api.annotation.download_batch(dataset.id, image_ids)

                for image, ann in zip(img_batch, ann_batch):
                    update_classes_distribution(stats, image, ann)

        aggregate_calculations(stats)

    else:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)
        for dataset in project_fs:
            for image in dataset:
                update_classes_distribution(stats, image)

        aggregate_calculations(stats)

    return stats


stats = classes_distribution(project_id=PROJECT_ID)

print()

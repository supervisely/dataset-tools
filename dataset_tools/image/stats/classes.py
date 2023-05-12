import os
import numpy as np
import random

import itertools
from collections import defaultdict

from typing import Dict

import supervisely as sly

BG_COLOR = [0, 0, 0]


class ImgClassesDistribution:
    """
    Important fields of modified stats dict:
        "class_names": [],
        "images_count": [],
        "image_counts_filter_by_id": [],
        "objects_count": [],
        "object_counts_filter_by_id": [],
        "avg_nonzero_area": [],
        "avg_nonzero_count": [],
    """

    @staticmethod
    def prepare_data(stats: Dict, meta):
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

        stats["class_names"] = class_names
        stats["class_indices_colors"] = class_indices_colors
        stats["_name_to_index"] = _name_to_index

        stats["sum_class_area_per_image"] = [0] * len(class_names)
        stats["sum_class_count_per_image"] = [0] * len(class_names)
        stats["count_images_with_class"] = [0] * len(class_names)

        stats["image_counts_filter_by_id"] = [[] for _ in class_names]
        stats["object_counts_filter_by_id"] = [[] for _ in class_names]

    @staticmethod
    def update(stats: Dict, image_info, ann_info, meta, *args, **kwargs):
        ann_json = ann_info.annotation
        ann_objects = [(obj["id"], obj["classTitle"]) for obj in ann_json["objects"]]

        ann = sly.Annotation.from_json(ann_json, meta)

        render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
        render_idx_rgb[:] = BG_COLOR

        ann.draw_class_idx_rgb(render_idx_rgb, stats["_name_to_index"])

        stat_area = sly.Annotation.stat_area(
            render_idx_rgb, stats["class_names"], stats["class_indices_colors"]
        )
        stat_count = ann.stat_class_count(stats["class_names"])

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        for idx, class_name in enumerate(stats["class_names"]):
            cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
            cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0

            stats["sum_class_area_per_image"][idx] += cur_area
            stats["sum_class_count_per_image"][idx] += cur_count
            stats["count_images_with_class"][idx] += 1 if stat_count[class_name] > 0 else 0

            if class_name == "unlabeled":
                continue
            if stat_count[class_name] > 0:
                stats["image_counts_filter_by_id"][idx].append(image_info.id)
            if stat_count[class_name] > 0:
                obj_ids = [obj[0] for obj in ann_objects if obj[1] == class_name]
                stats["object_counts_filter_by_id"][idx].extend(obj_ids)

        ann = sly.Annotation.from_json(ann_json, meta)

        stats["images_count"] = stats["count_images_with_class"]
        stats["objects_count"] = stats["sum_class_count_per_image"]

    @staticmethod
    def aggregate_calculations(stats: Dict):
        with np.errstate(divide="ignore"):
            avg_nonzero_area = np.divide(
                stats["sum_class_area_per_image"],
                stats["count_images_with_class"],
            )
            avg_nonzero_count = np.divide(
                stats["sum_class_count_per_image"],
                stats["count_images_with_class"],
            )

        avg_nonzero_area = np.where(np.isnan(avg_nonzero_area), None, avg_nonzero_area)
        avg_nonzero_count = np.where(np.isnan(avg_nonzero_count), None, avg_nonzero_count)

        stats["avg_nonzero_area"] = avg_nonzero_area.tolist()
        stats["avg_nonzero_count"] = avg_nonzero_count.tolist()


class ImgClassesCooccurence:
    """
    Important fields of modified stats dict:
        "class_names": [],
        "counters": [],
        "pd_data": [],
    """

    @staticmethod
    def prepare_data(stats: Dict, meta):
        class_names = [cls.name for cls in meta.obj_classes]
        counters = defaultdict(list)
        stats["class_names"] = class_names
        stats["counters"] = counters

    @staticmethod
    def update(stats: Dict, image_info, ann_info, meta, current_dataset):
        ann_json = ann_info.annotation
        ann = sly.Annotation.from_json(ann_json, meta)

        classes_on_image = set()
        for label in ann.labels:
            classes_on_image.add(label.obj_class.name)

        all_pairs = set(
            frozenset(pair) for pair in itertools.product(classes_on_image, classes_on_image)
        )
        for p in all_pairs:
            stats["counters"][p].append((image_info, current_dataset))

    @staticmethod
    def aggregate_calculations(stats: Dict):
        pd_data = []
        class_names = stats["class_names"]
        columns = ["name", *class_names]
        for cls_name1 in class_names:
            cur_row = [cls_name1]
            for cls_name2 in class_names:
                key = str(frozenset([cls_name1, cls_name2]))
                imgs_cnt = len(stats["counters"][key])
                cur_row.append(imgs_cnt)
            pd_data.append(cur_row)

        pd_data[:0] = [columns]
        stats["pd_data"] = pd_data

import itertools
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import dataframe_image as dfi
import numpy as np
import pandas as pd
import supervisely as sly

UNLABELED_COLOR = [0, 0, 0]


class ClassBalance:
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

    def __init__(self, project_meta: sly.ProjectMeta) -> None:
        self._meta = project_meta
        self._stats = {}

        self._class_names = ["unlabeled"]
        class_colors = [UNLABELED_COLOR]
        class_indices_colors = [UNLABELED_COLOR]
        self._name_to_index = {}
        for idx, obj_class in enumerate(self._meta.obj_classes):
            self._class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            class_index = idx + 1
            class_indices_colors.append([class_index, class_index, class_index])
            self._name_to_index[obj_class.name] = class_index

        self._stats["class_names"] = self._class_names
        self._stats["class_indices_colors"] = class_indices_colors
        self._stats["_name_to_index"] = self._name_to_index

        self._stats["sum_class_area_per_image"] = [0] * len(self._class_names)
        self._stats["objects_count"] = [0] * len(self._class_names)
        self._stats["images_count"] = [0] * len(self._class_names)

        self._stats["image_counts_filter_by_id"] = [[] for _ in self._class_names]
        # self._stats["object_counts_filter_by_id"] = [[] for _ in self._class_names]

        self._stats["avg_nonzero_area"] = [None] * len(self._class_names)
        self._stats["avg_nonzero_count"] = [None] * len(self._class_names)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
        render_idx_rgb[:] = UNLABELED_COLOR

        ann.draw_class_idx_rgb(render_idx_rgb, self._stats["_name_to_index"])

        stat_area = sly.Annotation.stat_area(
            render_idx_rgb, self._stats["class_names"], self._stats["class_indices_colors"]
        )
        stat_count = ann.stat_class_count(self._stats["class_names"])

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        for idx, class_name in enumerate(self._stats["class_names"]):
            cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
            cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0

            self._stats["sum_class_area_per_image"][idx] += cur_area
            self._stats["objects_count"][idx] += cur_count
            self._stats["images_count"][idx] += 1 if stat_count[class_name] > 0 else 0

            if self._stats["images_count"][idx] > 0:
                self._stats["avg_nonzero_area"][idx] = (
                    self._stats["sum_class_area_per_image"][idx] / self._stats["images_count"][idx]
                )
                self._stats["avg_nonzero_count"][idx] = (
                    self._stats["objects_count"][idx] / self._stats["images_count"][idx]
                )

            if class_name == "unlabeled":
                continue
                # if len(ann.labels) == 0:  # and stat_count["total"] == 0:
                #     self._stats["image_counts_filter_by_id"][idx].append(image.id)
            elif stat_count[class_name] > 0:
                self._stats["image_counts_filter_by_id"][idx].append(image.id)

            # TODO: implement later
            # if stat_count[class_name] > 0:
            # obj_ids = [obj[0] for obj in ann_objects if obj[1] == class_name]
            # self._stats["object_counts_filter_by_id"][idx].extend(obj_ids)

    def to_json(self) -> dict:
        columns = ["class", "images", "objects", "avg count per image", "avg area per image"]
        rows = []
        for name, idx in self._name_to_index.items():
            rows.append(
                [
                    name,
                    self._stats["images_count"][idx],
                    self._stats["objects_count"][idx],
                    round(self._stats["avg_nonzero_count"][idx], 2),
                    round(self._stats["avg_nonzero_area"][idx], 2),
                ]
            )

        colomns_options = [None] * len(columns)
        colomns_options[0] = {"type": "class"}
        colomns_options[1] = {"maxValue": max(self._stats["images_count"])}
        colomns_options[2] = {"maxValue": max(self._stats["objects_count"])}
        colomns_options[3] = {"maxValue": round(max(self._stats["avg_nonzero_count"]), 2)}
        colomns_options[4] = {
            "postfix": "%",
            "maxValue": round(max(self._stats["avg_nonzero_area"]), 2),
        }
        options = {"fixColumns": 1}

        res = {
            "columns": columns,
            "data": rows,
            "referencesRow": self._stats["image_counts_filter_by_id"],
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_pandas(self) -> pd.DataFrame:
        json = self.to_json()
        table = pd.DataFrame(data=json["data"], columns=json["columns"])
        return table

    def to_image(self, path):
        table = self.to_pandas()
        table.dfi.export(path)


# Max backup
# # Classes (grouped bar chart)
# classes_stats = {}


# def update_classes_stats(stats: dict, image: sly.ImageInfo, ann: sly.Annotation):
#     for key in ["objects", "images", "_temp"]:
#         if key not in stats:
#             stats[key] = {}

#     objects, images, temp = stats["objects"], stats["images"], stats["_temp"]
#     if "total_images" not in temp:
#         temp["total_images"] += 1  # total = with and without specific object on image

#     # avg count per image
#     # 5 + 7 + 8 + 12 = 32 / 4 = 8

#     class_flag = {}  # increment only once
#     for label in ann.labels:
#         name = label.obj_class.name
#         if name not in class_flag:
#             class_flag[name] = True
#             images[name] += 1
#         if name not in objects:
#             objects[name]["total"] = 0
#         objects[name]["total"] += 1
#         objects[name]["avg_num_per_image"] = objects[name]["total"] / temp["images_count"]


# def get_basic_classes_stats(project_id):
#     stats = {}
#     project_info = api.project.get_info_by_id(project_id)
#     project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
#     pbar = tqdm(total=project_info.items_count)
#     for dataset in api.dataset.get_list(project_id):
#         for batch in api.image.get_list_generator(dataset.id, batch_size=100):
#             image_ids = [image.id for image in batch]
#             anns = api.annotation.download_json_batch(dataset.id, image_ids)
#             # anns = [sly.Annotation.from_json(j, project_meta) for j in jann]
#             for image, jann in zip(batch, anns):
#                 ann = sly.Annotation.from_json(jann, project_meta)
#                 update_classes_stats()
#     pbar.close()


# res = get_basic_classes_stats(project_id)

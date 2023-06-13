from typing import Dict

import numpy as np

import supervisely as sly
from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]


class ClassBalance(BaseStats):
    """
    Columns:
        Class
        Images
        Objects
        Avg count per image
        Avg area per image
    """

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False, stat_cache: dict = None) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force
        self._stat_cache = stat_cache

        self._class_names = ["unlabeled"]
        class_colors = [UNLABELED_COLOR]
        class_indices_colors = [UNLABELED_COLOR]

        # self._class_names = []
        # class_colors = []
        # class_indices_colors = []
        self._name_to_index = {}
        for idx, obj_class in enumerate(self._meta.obj_classes):
            self._class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            cls_idx = idx + 1
            # class_indices_colors.append([idx, idx, idx])
            class_indices_colors.append([cls_idx, cls_idx, cls_idx])
            self._name_to_index[obj_class.name] = cls_idx

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
        cur_class_names = ["unlabeled"]
        cur_class_colors = [UNLABELED_COLOR]
        classname_to_index = {}

        for label in ann.labels:
            if label.obj_class.name not in cur_class_names:
                cur_class_names.append(label.obj_class.name)
                class_index = len(cur_class_colors) + 1
                cur_class_colors.append([class_index, class_index, class_index])
                classname_to_index[label.obj_class.name] = class_index

        if self._stat_cache is not None and image.id in self._stat_cache:
            stat_area = self._stat_cache[image.id]["stat_area"]
        else:
            render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
            render_idx_rgb[:] = UNLABELED_COLOR
            ann.draw_class_idx_rgb(render_idx_rgb, classname_to_index)
            stat_area = sly.Annotation.stat_area(render_idx_rgb, cur_class_names, cur_class_colors)
            if self._stat_cache is not None:
                if image.id in self._stat_cache:
                    self._stat_cache[image.id]["stat_area"] = stat_area
                else:
                    self._stat_cache[image.id] = {"stat_area": stat_area}
        stat_count = ann.stat_class_count(cur_class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        for idx, class_name in enumerate(self._stats["class_names"]):
            if class_name not in cur_class_names:
                cur_area = 0
                cur_count = 0
                self._stats["images_count"][idx] += 0
            else:
                cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
                cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
                self._stats["images_count"][idx] += 1 if stat_count[class_name] > 0 else 0

            self._stats["sum_class_area_per_image"][idx] += cur_area
            self._stats["objects_count"][idx] += cur_count

            if self._stats["images_count"][idx] > 0:
                self._stats["avg_nonzero_area"][idx] = (
                    self._stats["sum_class_area_per_image"][idx] / self._stats["images_count"][idx]
                )
                self._stats["avg_nonzero_count"][idx] = (
                    self._stats["objects_count"][idx] / self._stats["images_count"][idx]
                )

            if class_name == "unlabeled":
                continue
            elif class_name in cur_class_names:
                if stat_count[class_name] > 0:
                    self._stats["image_counts_filter_by_id"][idx].append(image.id)

            # TODO: implement later
            # if stat_count[class_name] > 0:
            # obj_ids = [obj[0] for obj in ann_objects if obj[1] == class_name]
            # self._stats["object_counts_filter_by_id"][idx].extend(obj_ids)

    def to_json(self) -> Dict:
        columns = [
            "Class",
            "Images",
            "Objects",
            "Count on image",
            "Area on image",
        ]
        rows = []
        for name, idx in self._name_to_index.items():
            rows.append(
                [
                    name,
                    self._stats["images_count"][idx],
                    self._stats["objects_count"][idx],
                    round(self._stats["avg_nonzero_count"][idx] or 0, 2),
                    round(self._stats["avg_nonzero_area"][idx] or 0, 2),
                ]
            )
        notnonecount = [item for item in self._stats["avg_nonzero_count"] if item is not None]
        notnonearea = [item for item in self._stats["avg_nonzero_area"] if item is not None]
        colomns_options = [None] * len(columns)
        colomns_options[0] = {"type": "class"}
        colomns_options[1] = {
            "maxValue": max(self._stats["images_count"]),
            "tooltip": "Number of images with at least one object of corresponding class",
        }
        colomns_options[2] = {
            "maxValue": max(self._stats["objects_count"]),
            "tooltip": "Number of objects of corresponding class in the project",
        }
        colomns_options[3] = {
            "maxValue": round(max(notnonecount), 2),
            "subtitle": "average",
            "tooltip": "Average number of objects of corresponding class on the image. Images without such objects are not taking into account",
        }
        colomns_options[4] = {
            "postfix": "%",
            "maxValue": round(max(notnonearea), 2),
            "subtitle": "average",
            "tooltip": "Average image area of corresponding class. Images without such objects are not taking into account",
        }
        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "desc"},
            "pageSize": len(self._class_names),
        }  # asc

        res = {
            "columns": columns,
            "data": rows,
            "referencesRow": self._stats["image_counts_filter_by_id"][1:],
            "options": options,
            "columnsOptions": colomns_options,
        }
        return res

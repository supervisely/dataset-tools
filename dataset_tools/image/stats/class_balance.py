import multiprocessing
import random
from typing import Dict, List

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]
REFERENCES_LIMIT = 1000


class ClassBalance(BaseStats):
    """
    Columns:
        Class
        Images
        Objects
        Avg count per image
        Avg area per image
    """

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_stats,
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force
        self._stat_cache = stat_cache

        self.references_probabilities = {}
        for cls in project_stats["images"]["objectClasses"]:
            self.references_probabilities[cls["objectClass"]["name"]] = (
                REFERENCES_LIMIT / cls["total"] if cls["total"] != 0 else 1
            )

        self.class_names = ["unlabeled"]
        class_colors = [UNLABELED_COLOR]
        class_indices_colors = [UNLABELED_COLOR]

        self._name_to_index = {}
        for idx, obj_class in enumerate(self._meta.obj_classes):
            self.class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            cls_idx = idx + 1
            class_indices_colors.append([cls_idx, cls_idx, cls_idx])
            self._name_to_index[obj_class.name] = cls_idx

        self.class_indices_colors = class_indices_colors

        self.sum_class_area_per_image = [0] * len(self.class_names)
        self.objects_count = [0] * len(self.class_names)
        self.images_count = [0] * len(self.class_names)

        self.image_counts_filter_by_id = [[] for _ in self.class_names]
        # self.dataset_counts_filter_by_id = [{} for _ in self.class_names]
        # self.ds_position = [0 for _ in self.class_names]
        # self.accum_ids = [set() for _ in self.class_names]

        self.avg_nonzero_area = [None] * len(self.class_names)
        self.avg_nonzero_count = [None] * len(self.class_names)

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
            masks = []
            for cls in cur_class_names[1:]:
                render_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)

                class_labels = [label for label in ann.labels if label.obj_class.name == cls]
                clann = ann.clone(labels=class_labels)

                clann.draw(render_rgb, [1, 1, 1])
                masks.append(render_rgb)

            if len(masks) == 0:
                stat_area = {"unlabeled": 100}

            else:
                bitmasks1channel = [mask[:, :, 0] for mask in masks]
                stacked_masks = np.stack(bitmasks1channel, axis=2)
                total_area = stacked_masks.shape[0] * stacked_masks.shape[1]
                mask_areas = (np.sum(stacked_masks, axis=(0, 1)) / total_area) * 100

                mask_areas = np.insert(mask_areas, 0, self.calc_unlabeled_area_in(masks))
                stat_area = {cls: area for cls, area in zip(cur_class_names, mask_areas.tolist())}

                if self._stat_cache is not None:
                    if image.id in self._stat_cache:
                        self._stat_cache[image.id]["stat_area"] = stat_area
                    else:
                        self._stat_cache[image.id] = {"stat_area": stat_area}

        stat_count = ann.stat_class_count(cur_class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        for idx, class_name in enumerate(self.class_names):
            if class_name not in cur_class_names:
                cur_area = 0
                cur_count = 0
                self.images_count[idx] += 0
            else:
                cur_area = stat_area.get(class_name, 0)# if not np.isnan(stat_area[class_name]) else 0
                cur_count = stat_count.get(class_name, 0) #] if not np.isnan(stat_count[class_name]) else 0
                self.images_count[idx] += 1 if cur_count > 0 else 0

            self.sum_class_area_per_image[idx] += cur_area
            self.objects_count[idx] += cur_count

            if self.images_count[idx] > 0:
                self.avg_nonzero_area[idx] = (
                    self.sum_class_area_per_image[idx] / self.images_count[idx]
                )
                self.avg_nonzero_count[idx] = self.objects_count[idx] / self.images_count[idx]

            if class_name in cur_class_names[1:]:
                if (
                    stat_count[class_name] > 0
                    and random.random() < self.references_probabilities[class_name]
                ):
                    self.image_counts_filter_by_id[idx].append(image.id)

                    # if image.dataset_id not in self.accum_ids[idx]:
                    #     self.dataset_counts_filter_by_id[idx].update(
                    #         {self.ds_position[idx]: image.dataset_id}
                    #     )
                    #     self.accum_ids[idx].add(image.dataset_id)
                    #     # self.accum_ids[idx] = list(set(self.accum_ids[idx]))
                    # self.ds_position[idx] += 1

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
                    self.images_count[idx],
                    self.objects_count[idx],
                    round(self.avg_nonzero_count[idx] or 0, 2),
                    round(self.avg_nonzero_area[idx] or 0, 2),
                ]
            )
        notnonecount = [item for item in self.avg_nonzero_count if item is not None]
        notnonearea = [item for item in self.avg_nonzero_area if item is not None]

        colomns_options = [None] * len(columns)
        colomns_options[0] = {"type": "class"}
        colomns_options[1] = {
            "maxValue": max(self.images_count),
            "tooltip": "Number of images with at least one object of corresponding class",
        }
        colomns_options[2] = {
            "maxValue": max(self.objects_count),
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
            "pageSize": 10,
        }  # asc

        res = {
            "columns": columns,
            "data": rows,
            "referencesRow": self.image_counts_filter_by_id[1:],
            "options": options,
            "columnsOptions": colomns_options,
        }
        return res

    def parallel_update(self, images, annotations, num_processes):
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(self.process_image, [(img, ann) for img, ann in zip(images, annotations)])
        pool.close()
        pool.join()

    def process_image(self, args):
        image, annotation = args
        self.update(image, annotation)

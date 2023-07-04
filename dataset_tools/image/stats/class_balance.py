from typing import Dict, List

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

    def __init__(
        self, project_meta: sly.ProjectMeta, force: bool = False, stat_cache: dict = None
    ) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force
        self._stat_cache = stat_cache

        self._class_names = ["unlabeled"]
        class_colors = [UNLABELED_COLOR]
        class_indices_colors = [UNLABELED_COLOR]

        self._name_to_index = {}
        for idx, obj_class in enumerate(self._meta.obj_classes):
            self._class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            cls_idx = idx + 1
            class_indices_colors.append([cls_idx, cls_idx, cls_idx])
            self._name_to_index[obj_class.name] = cls_idx

        self._stats["class_names"] = self._class_names
        self._stats["class_indices_colors"] = class_indices_colors
        self._stats["_name_to_index"] = self._name_to_index

        self._stats["sum_class_area_per_image"] = [0] * len(self._class_names)
        self._stats["objects_count"] = [0] * len(self._class_names)
        self._stats["images_count"] = [0] * len(self._class_names)

        self._stats["image_counts_filter_by_id"] = [[] for _ in self._class_names]

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
            masks = []
            for cls in self._stats["class_names"]:
                if cls != "unlabeled":
                    render_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
                    render_rgb[:] = UNLABELED_COLOR
                    class_labels = [label for label in ann.labels if label.obj_class.name == cls]
                    clann = ann.clone(labels=class_labels)
                    # if not class_labels:
                    #     clanns.append(cls)  # List[Union[str, sly.Annotation]]
                    # else:
                    #     clanns.append(clann)  # List[Union[str, sly.Annotation]]
                    clann.draw(render_rgb, [1, 1, 1])
                    masks.append(render_rgb)

            stat_area = {}

            bitmasks1channel = [mask[:, :, 0] for mask in masks]
            stacked_masks = np.stack(bitmasks1channel, axis=2)

            total_area = stacked_masks.shape[0] * stacked_masks.shape[1]
            mask_areas = (np.sum(stacked_masks, axis=(0, 1)) / total_area) * 100

            mask_areas = np.insert(mask_areas, 0, self.calc_unlabeled_area_in(masks))
            stat_area = {
                cls: area for cls, area in zip(self._stats["class_names"], mask_areas.tolist())
            }

            # if overlapping:
            #     stat_area = {}
            #     stat_area["unlabeled"] = self.calc_unlabeled_area_in_overlapped(masks)

            #     grouped_equal_masks = self.group_equal_masks(masks)
            #     unique_ann_dict = {}

            #     for idx, clann in enumerate(clanns):
            #         if isinstance(clann, str):
            #             cls_name = clann
            #             stat_area[cls_name] = 0
            #             continue

            #         assert (
            #             len(set([label.obj_class.name for label in clann.labels])) == 1
            #         ), "'clann' annotation should contain labels from single class"
            #         cls_name = clann.labels[0].obj_class.name

            #         same_cls = unique_ann_dict.get(grouped_equal_masks[idx])
            #         unique_ann_dict[grouped_equal_masks[idx]] = cls_name
            #         if same_cls is not None:
            #             stat_area[cls_name] = stat_area[same_cls]
            #             continue

            #         render_idx_rgb = np.zeros(clann.img_size + (3,), dtype=np.uint8)
            #         render_idx_rgb[:] = UNLABELED_COLOR

            #         clann.draw_class_idx_rgb(render_idx_rgb, classname_to_index)
            #         tmp_stat_area = sly.Annotation.stat_area(
            #             render_idx_rgb, cur_class_names, cur_class_colors
            #         )

            #         stat_area[cls_name] = tmp_stat_area[cls_name]
            # else:
            # render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
            # render_idx_rgb[:] = UNLABELED_COLOR
            # ann.draw_class_idx_rgb(render_idx_rgb, classname_to_index)
            # stat_area = sly.Annotation.stat_area(render_idx_rgb, cur_class_names, cur_class_colors)

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
            "pageSize": 10,
        }  # asc

        res = {
            "columns": columns,
            "data": rows,
            "referencesRow": self._stats["image_counts_filter_by_id"][1:],
            "options": options,
            "columnsOptions": colomns_options,
        }
        return res

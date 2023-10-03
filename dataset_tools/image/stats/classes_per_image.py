import random
from typing import Dict, List

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]
CLASSES_CNT_LIMIT = 100

MAX_SIZE_OBJECT_SIZES_BYTES = 1e7
SHRINKAGE_COEF = 0.1

class ClassesPerImage(BaseStats):
    """
    Columns:
        Image
        Dataset
        Height
        Width
        Unlabeled
        Class1 objects count
        Class1 covered area (%)
        Class2 objects count
        Class2 covered area (%)
        etc.
    """

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_stats: dict,
        datasets: List[sly.DatasetInfo] = None,
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force
        self._stat_cache = stat_cache
        self.project_stats = project_stats

        self._dataset_id_to_name = None
        if datasets is not None:
            self._dataset_id_to_name = {ds.id: ds.name for ds in datasets}

        self._class_names = ["unlabeled"]
        self._class_indices_colors = [UNLABELED_COLOR]
        self._classname_to_index = {}

        for idx, obj_class in enumerate(self._meta.obj_classes):
            if idx >= CLASSES_CNT_LIMIT:
                sly.logger.warn(
                    f"{self.__class__.__name__}: will use first {CLASSES_CNT_LIMIT} classes."
                )
                break
            self._class_names.append(obj_class.name)
            class_index = idx + 1
            self._class_indices_colors.append([class_index, class_index, class_index])
            self._classname_to_index[obj_class.name] = class_index

        self._stats["data"] = []
        self._referencesRow = []

        total = self.project_stats["images"]["total"]['imagesInDataset'] * (len(self.project_stats['images']["objectClasses"])+5)
        self.update_freq = 1       
        if total  > MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF:
            self.update_freq = MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF / total  


    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        if self.update_freq >= random.random():

            cur_class_names = ["unlabeled"]
            cur_class_colors = [UNLABELED_COLOR]
            classname_to_index = {}

            for label in ann.labels:
                if label.obj_class.name not in cur_class_names:
                    cur_class_names.append(label.obj_class.name)
                    class_index = len(cur_class_colors)
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

            table_row = []

            table_row.append(image.name)

            if self._dataset_id_to_name is not None:
                table_row.append(self._dataset_id_to_name[image.dataset_id])
            area_unl = stat_area.get("unlabeled", 0) # if not np.isnan(stat_area["unlabeled"]) else 0
            table_row.extend(
                [
                    image.height,  # stat_area["height"],
                    image.width,  # stat_area["width"],
                    round(area_unl, 2) if area_unl != 0 else 0,
                ]
            )
            for class_name in self._class_names[1:]:
                # if class_name == "unlabeled":
                #     continue
                if class_name not in cur_class_names:
                    cur_area = 0
                    cur_count = 0
                else:
                    cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
                    cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
                table_row.append(cur_count)
                table_row.append(round(cur_area, 2) if cur_area != 0 else 0)

            self._stats["data"].append(table_row)
            self._referencesRow.append([image.id])

    def to_json(self) -> Dict:
        if self._dataset_id_to_name is not None:
            columns = ["Image", "Split", "Height", "Width", "Unlabeled"]
        else:
            columns = ["Image", "Height", "Width", "Unlabeled"]

        columns_options = [None] * len(columns)

        if self._dataset_id_to_name is not None:
            columns_options[columns.index("Split")] = {
                "subtitle": "folder name",
            }
        columns_options[columns.index("Height")] = {
            "postfix": "px",
        }
        columns_options[columns.index("Width")] = {
            "postfix": "px",
        }
        columns_options[columns.index("Unlabeled")] = {
            "subtitle": "area",
            "postfix": "%",
        }

        for class_name in self._class_names:
            if class_name == "unlabeled":
                continue
            columns_options.append({"subtitle": "objects count"})
            columns_options.append({"subtitle": "covered area", "postfix": "%"})
            columns.extend([class_name] * 2)

        options = {"fixColumns": 1}
        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": self._stats["data"],
            "options": options,
            "referencesRow": self._referencesRow,
        }
        return res

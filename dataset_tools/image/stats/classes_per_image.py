from typing import List, Dict

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]


class ClassesPerImage(BaseStats):
    """
    Columns:
        Image
        Dataset
        Height
        Width
        # Channels
        Unlabeled
        Class1 objects count
        Class1 covered area (%)
        Class2 objects count
        Class2 covered area (%)
        etc.
    """

    def __init__(
        self, project_meta: sly.ProjectMeta, datasets: List[sly.DatasetInfo] = None, force:bool = False
    ) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force
        
        self._dataset_id_to_name = None
        if datasets is not None:
            self._dataset_id_to_name = {ds.id: ds.name for ds in datasets}

        self._class_names = ["unlabeled"]
        self._class_indices_colors = [UNLABELED_COLOR]
        self._classname_to_index = {}

        for idx, obj_class in enumerate(self._meta.obj_classes):
            self._class_names.append(obj_class.name)
            class_index = idx + 1
            self._class_indices_colors.append([class_index, class_index, class_index])
            self._classname_to_index[obj_class.name] = class_index

        self._stats["data"] = []
        self._referencesRow = []

    def update(self, image_info: sly.ImageInfo, ann: sly.Annotation) -> None:
        render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
        render_idx_rgb[:] = UNLABELED_COLOR
        ann.draw_class_idx_rgb(render_idx_rgb, self._classname_to_index)
        stat_area = sly.Annotation.stat_area(
            render_idx_rgb, self._class_names, self._class_indices_colors
        )
        stat_count = ann.stat_class_count(self._class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        table_row = []

        table_row.append(image_info.name)

        if self._dataset_id_to_name is not None:
            table_row.append(self._dataset_id_to_name[image_info.dataset_id])
        area_unl = stat_area["unlabeled"] if not np.isnan(stat_area["unlabeled"]) else 0
        table_row.extend(
            [
                stat_area["height"],
                stat_area["width"],
                # stat_area["channels"],
                round(area_unl, 2) if area_unl != 0 else 0,
            ]
        )
        for class_name in self._class_names:
            cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
            cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
            if class_name == "unlabeled":
                continue
            table_row.append(cur_count)
            table_row.append(round(cur_area, 2) if cur_area != 0 else 0)

        self._stats["data"].append(table_row)
        self._referencesRow.append([image_info.id])

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

        for obj_class in self._meta.obj_classes:
            columns_options.append({"subtitle": "objects count"})
            columns_options.append({"subtitle": "covered area", "postfix": "%"})
            columns.extend([obj_class.name] * 2)

        options = {"fixColumns": 1}
        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": self._stats["data"],
            "options": options,
            "referencesRow": self._referencesRow,
        }
        return res

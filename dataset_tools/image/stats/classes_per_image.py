from typing import Dict, List

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]
CLASSES_CNT_LIMIT = 100


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
        datasets: List[sly.DatasetInfo] = None,
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force
        self._stat_cache = stat_cache

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

    def update(self, image_info: sly.ImageInfo, ann: sly.Annotation) -> None:
        cur_class_names = ["unlabeled"]
        cur_class_colors = [UNLABELED_COLOR]
        classname_to_index = {}

        for label in ann.labels:
            if label.obj_class.name not in cur_class_names:
                cur_class_names.append(label.obj_class.name)
                class_index = len(cur_class_colors)
                cur_class_colors.append([class_index, class_index, class_index])
                classname_to_index[label.obj_class.name] = class_index

        if self._stat_cache is not None and image_info.id in self._stat_cache:
            stat_area = self._stat_cache[image_info.id]["stat_area"]
        else:
            masks = []
            for cls in self._class_names:
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
            stat_area = {cls: area for cls, area in zip(self._class_names, mask_areas.tolist())}

            # overlapping = self.check_overlap(masks)

            # if overlapping:
            #     stat_area = {}
            #     stat_area["unlabeled"] = self.calc_unlabeled_area_in(masks)

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
            #     render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
            #     render_idx_rgb[:] = UNLABELED_COLOR
            #     ann.draw_class_idx_rgb(render_idx_rgb, classname_to_index)
            #     stat_area = sly.Annotation.stat_area(
            #         render_idx_rgb, cur_class_names, cur_class_colors
            #     )

            if self._stat_cache is not None:
                if image_info.id in self._stat_cache:
                    self._stat_cache[image_info.id]["stat_area"] = stat_area
                else:
                    self._stat_cache[image_info.id] = {"stat_area": stat_area}

        stat_count = ann.stat_class_count(cur_class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        table_row = []

        table_row.append(image_info.name)

        if self._dataset_id_to_name is not None:
            table_row.append(self._dataset_id_to_name[image_info.dataset_id])
        area_unl = stat_area["unlabeled"] if not np.isnan(stat_area["unlabeled"]) else 0
        table_row.extend(
            [
                image_info.height,  # stat_area["height"],
                image_info.width,  # stat_area["width"],
                round(area_unl, 2) if area_unl != 0 else 0,
            ]
        )
        for class_name in self._class_names:
            if class_name == "unlabeled":
                continue
            if class_name not in cur_class_names:
                cur_area = 0
                cur_count = 0
            else:
                cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
                cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
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

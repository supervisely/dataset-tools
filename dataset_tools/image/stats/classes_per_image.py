import copy
import random
from typing import Dict, List, Optional

import numpy as np
import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo

from dataset_tools.image.stats.basestats import BaseStats
from collections import defaultdict

UNLABELED_COLOR = [0, 0, 0]
CLASSES_CNT_LIMIT = 200

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
        cls_prevs_tags: list = [],
        sly_tag_split: dict = {},
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        self.project_stats = project_stats
        self.datasets = datasets
        self.force = force
        self._stat_cache = stat_cache

        self._cls_prevs_tags = set(cls_prevs_tags)
        self._sly_tag_split = sly_tag_split

        # self._columns = ["Image"]
        self._columns = []

        self._stats = {}

        self._dataset_id_to_name = None
        if datasets is not None:
            self._dataset_id_to_name = self._get_aggregated_names(datasets)
            # self._dataset_id_to_name = {ds.id: ds.name for ds in datasets}
            # self._columns.append("Split")

        # start_columns_len = len(self._columns)

        self._tag_to_position = {}
        self._sly_tag_split_len = 0

        for curr_split, tag_split_list in self._sly_tag_split.items():
            if curr_split == "__POSTTEXT__" or curr_split == "__PRETEXT__":
                continue

            intersection_tags = list(set(tag_split_list) - self._cls_prevs_tags)
            if len(intersection_tags) == 0:
                continue

            self._columns.append(curr_split)
            self._sly_tag_split_len += 1

            for tag in tag_split_list:
                self._tag_to_position[tag] = self._sly_tag_split_len + 1  # + start_columns_len

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
        self._references = []

        total = self.project_stats["images"]["total"]["imagesInDataset"] * (
            len(self.project_stats["images"]["objectClasses"]) + 5
        )
        self.update_freq = 1
        if total > MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF:
            self.update_freq = MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF / total

        # new
        self._splits = self._get_aggregated_names(datasets)
        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes}
        self._data_dict = {}

    def clean(self):
        self.__init__(
            self._meta,
            self.project_stats,
            self.datasets,
            self._cls_prevs_tags,
            self._sly_tag_split,
            self.force,
            self._stat_cache,
        )

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            pass  # keep unlabeled images

        counts = {class_id: 0 for class_id in self._class_ids}
        areas = {class_id: 0 for class_id in self._class_ids}
        # bboxes = {class_id: [] for class_id in self._class_ids}

        row_dict = {
            "image": image.name,
            "dataset": self._splits[image.dataset_id],
            "height": image.height,
            "width": image.width,
        }

        image_area = image.width * image.height

        for figure in figures:
            counts[figure.class_id] += 1
            area_percent = float(figure.area) / image_area * 100
            areas[figure.class_id] += area_percent
            # bboxes[figure.class_id].append(figure.geometry_meta["bbox"])

        row_dict["classes"] = dict()
        for class_id in self._class_ids:
            # canvas = np.zeros((image.height, image.width), dtype=int)
            # unlabeled_cls_area = self._count_unlabeled_area(canvas, bboxes[class_id])
            # areas[class_id] = (1 - unlabeled_cls_area) * 100
            row_dict["classes"][class_id] = [
                counts[class_id],
                round(areas[class_id], 2),
            ]

        self._data_dict[image.id] = row_dict

    def to_json2(self):

        columns = ["Image", "Dataset", "Height", "Width"]  # , "Unlabeled"]
        columns_options = [None] * len(columns)

        fkey = next(iter(self._data_dict))
        cls_ids = self._data_dict[fkey]["classes"]
        for cls_id in cls_ids:
            subcols = [self._class_ids[cls_id]] * 2
            columns.extend(subcols)

        data = []
        references = []

        for image_id, row in self._data_dict.items():
            data.append(
                [
                    row["image"],
                    row["dataset"],
                    row["height"],
                    row["width"],
                ]
                + [value for x in cls_ids for value in row["classes"][x]]
            )
            references.append([image_id])

        columns_options[columns.index("Dataset")] = {
            "subtitle": "folder name",
        }
        columns_options[columns.index("Height")] = {
            "postfix": "px",
        }
        columns_options[columns.index("Width")] = {
            "postfix": "px",
        }
        # columns_options[columns.index("Unlabeled")] = {
        #     "subtitle": "area",
        #     "postfix": "%",
        # }

        for _ in self._class_ids:
            columns_options.append({"subtitle": "objects count"})
            columns_options.append({"subtitle": "covered area", "postfix": "%"})

        options = {"fixColumns": 1}

        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": data,
            "options": options,
            "referencesRow": references,
        }
        return res

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
                    stat_area = {
                        cls: area for cls, area in zip(cur_class_names, mask_areas.tolist())
                    }

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

            if self._sly_tag_split_len > 0:
                table_row.extend([None] * self._sly_tag_split_len)
                for tag in ann.img_tags:
                    tag_position = self._tag_to_position.get(tag.name)
                    if tag_position is not None:
                        table_row[tag_position] = tag.name

            area_unl = stat_area.get(
                "unlabeled", 0
            )  # if not np.isnan(stat_area["unlabeled"]) else 0
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
                    cur_count = (
                        stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
                    )
                table_row.append(cur_count)
                table_row.append(round(cur_area, 2) if cur_area != 0 else 0)

            self._stats["data"].append(table_row)
            self._references.append([image.id])

    def to_json(self) -> Dict:

        if self._dataset_id_to_name is not None:
            columns = ["Image", "Split"] + self._columns + ["Height", "Width", "Unlabeled"]
        else:
            columns = ["Image"] + self._columns + ["Height", "Width", "Unlabeled"]

        # self._columns.extend(["Height", "Width", "Unlabeled"])

        # columns = copy.deepcopy(self._columns)
        columns_options = [None] * len(columns)

        if self._dataset_id_to_name is not None:
            columns_options[columns.index("Split")] = {
                "subtitle": "folder name",
            }

        for curr_split in self._sly_tag_split.keys():
            if curr_split == "__POSTTEXT__" or curr_split == "__PRETEXT__":
                continue
            columns_options[columns.index(curr_split)] = {
                "subtitle": "tag split",
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
            "referencesRow": self._references,
        }
        return res

    def to_numpy_raw(self):
        return np.array(self._data_dict, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str):
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:

            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                fkey = next(iter(loaded_data))
                loaded_classes = set(loaded_data[fkey]["classes"])
                true_classes = set(self._class_ids)

                added = true_classes - loaded_classes
                removed = loaded_classes - true_classes

                if len(added) > 0 or len(removed) > 0:
                    for row in loaded_data.values():
                        for cls_id_new in added:
                            row["classes"][cls_id_new] = [0, 0]
                        for cls_id_rm in removed:
                            row["classes"].pop(cls_id_rm, None)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                self._data_dict.update(loaded_data)

    def _count_unlabeled_area(self, canvas, bounding_boxes):
        for bbox in bounding_boxes:
            sly.Rectangle(*bbox)
            y_min, x_min, y_max, x_max = bbox
            canvas[y_min:y_max, x_min:x_max] = 1
        return np.sum(canvas == 0) / canvas.size

    def _get_aggregated_names(self, datasets: List) -> Dict:
        id_to_name = {}
        id_to_info = {ds.id: ds for ds in datasets}
        for dataset in datasets:
            original_id = dataset.id
            dataset_name = dataset.name
            current = dataset
            while True:
                parent = current.parent_id
                if parent is None:
                    break
                current = id_to_info[parent]
                dataset_name = current.name + '/' + dataset_name
            id_to_name[original_id] = dataset_name
        return id_to_name

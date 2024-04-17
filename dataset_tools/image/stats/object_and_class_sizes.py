import math
import os
import random
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional

import numpy as np
import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import TreemapChart

from dataset_tools.image.stats.basestats import BaseStats

MAX_SIZE_OBJECT_SIZES_BYTES = 1e7
SHRINKAGE_COEF = 0.01

# LiteGeometry = namedtuple("LiteGeometry", ["__class__"])
LiteLabel = namedtuple(
    "LiteLabel",
    ["obj_class_name", "geometry_type", "geometry_to_bbox", "geometry_area"],
)
LiteAnnotation = namedtuple("LiteAnnotation", ["labels", "img_size"])


class ObjectSizes(BaseStats):
    """
    Columns:
        Object ID
        Class
        Dataset ID
        Image name
        Image size
        Height px
        Height %
        Width px
        Width %
        Area %
    """

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_stats,
        datasets: List[sly.DatasetInfo] = None,
        force: bool = False,
    ):
        self._meta = project_meta
        self.project_stats = project_stats
        self.datasets = datasets
        self.force = force

        self._dataset_id_to_name = None
        if datasets is not None:
            self._dataset_id_to_name = {ds.id: ds.name for ds in datasets}
        self._stats = []
        self._object_id = 1

        total_objects = self.project_stats["objects"]["total"]["objectsInDataset"]
        self.update_freq = 1
        if total_objects > MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF:
            self.update_freq = MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF / total_objects
        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes.items()}

        # new
        self._stats2 = {}
        self._stats2["data"] = []
        self._stats2["refs"] = []

    def clean(self):
        self.__init__(
            self._meta,
            self.project_stats,
            self.datasets,
            self.force,
        )

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            return

        image_height, image_width = image.height, image.width

        for figure in figures:
            if figure.geometry_type not in [
                sly.Bitmap.name(),
                sly.Rectangle.name(),
                sly.Polygon.name(),
                sly.GraphNodes.name(),
                sly.Point.name(),
            ]:
                continue

            object_id = self._object_id
            self._object_id += 1

            object_data = {
                "object_id": object_id,
                "class": self._class_ids[figure.class_id],
                "image_name": image.name,
            }

            if self._dataset_id_to_name:
                dataset_name = self._dataset_id_to_name[image.dataset_id]
                object_data["dataset_name"] = dataset_name

            object_data["image_size_hw"] = f"{image_height} x {image_width}"

            lite_label = LiteLabel(
                obj_class_name=self._class_ids[figure.class_id],
                geometry_type=figure.geometry_type,
                geometry_to_bbox=figure.bbox,
                geometry_area=figure.area,
            )
            object_data.update(calculate_obj_sizes(lite_label, image_height, image_width))

            object_data = list(object_data.values())

            # self._stats.append((object_data, [image.id]))
            self._stats2["data"].append(object_data)
            self._stats2["refs"].append([image.id])

    def to_json2(self):
        if not self._stats2["data"]:
            sly.logger.warning(
                "ObjectSizes: No stats were added in update() method, the result will be None."
            )
            return

        options = {
            "sort": {"columnIndex": 0, "order": "asc"},
        }

        columns = [
            "Object ID",
            "Class",
            "Image name",
            "Image size",
            "Height",
            "Height",
            "Width",
            "Width",
            "Area",
        ]

        columns_options = [
            {"tooltip": "ID of the object in instance"},
            {"type": "class"},
            {"subtitle": "click row to open"},
            {"subtitle": "height x width"},
            {"postfix": "px"},
            {"postfix": "%"},
            {"postfix": "px"},
            {"postfix": "%"},
            {"postfix": "%"},
        ]

        if self._dataset_id_to_name:
            columns.insert(3, "Split")
            columns_options.insert(
                3,
                {
                    "subtitle": "folder name",
                },
            )

        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": self._stats2["data"],
            "options": options,
            "referencesRow": self._stats2["refs"],
        }

        return res

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        if self.update_freq >= random.random():
            image_height, image_width = ann.img_size

            for label in ann.labels:
                if type(label.geometry) not in [sly.Bitmap, sly.Rectangle, sly.Polygon]:
                    continue

                object_id = self._object_id
                self._object_id += 1

                object_data = {
                    "object_id": object_id,
                    "class": label.obj_class.name,
                    "image_name": image.name,
                }

                if self._dataset_id_to_name:
                    dataset_name = self._dataset_id_to_name[image.dataset_id]
                    object_data["dataset_name"] = dataset_name

                object_data["image_size_hw"] = f"{image_height} x {image_width}"

                lite_label = LiteLabel(
                    obj_class_name=label.obj_class.name,
                    geometry_type=type(label.geometry),
                    geometry_to_bbox=label.geometry.to_bbox(),
                    geometry_area=label.geometry.area,
                )
                object_data.update(calculate_obj_sizes(lite_label, image_height, image_width))

                object_data = list(object_data.values())

                self._stats.append((object_data, [image.id]))

    def to_json(self) -> Dict:
        if not self._stats:
            sly.logger.warning(
                "ObjectSizes: No stats were added in update() method, the result will be None."
            )
            return

        options = {
            "sort": {"columnIndex": 0, "order": "asc"},
        }

        columns = [
            "Object ID",
            "Class",
            "Image name",
            "Image size",
            "Height",
            "Height",
            "Width",
            "Width",
            "Area",
        ]

        columns_options = [
            {"tooltip": "ID of the object in instance"},
            {"type": "class"},
            {"subtitle": "click row to open"},
            {"subtitle": "height x width"},
            {"postfix": "px"},
            {"postfix": "%"},
            {"postfix": "px"},
            {"postfix": "%"},
            {"postfix": "%"},
        ]

        if self._dataset_id_to_name:
            columns.insert(3, "Split")
            columns_options.insert(
                3,
                {
                    "subtitle": "folder name",
                },
            )

        data, references_row = zip(*self._stats)

        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": data,
            "options": options,
            "referencesRow": references_row,
        }

        return res

    def to_numpy_raw(self) -> np.ndarray:
        return np.array(self._stats2, dtype=object)
        # return np.array((self._data, self._references))

    # @sly.timeit
    def sew_chunks(self, chunks_dir, *args, **kwargs) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        res = []
        references = []

        def custom_key(path):
            # Split path and extract dataset ID and chunk ID
            parts = os.path.basename(path).split("_")
            return int(parts[2]), int(parts[1])

        # Sort paths by dataset ID and then by chunk ID
        sorted_files = sorted(files, key=custom_key)

        for file in sorted_files:
            loaded_data = np.load(file, allow_pickle=True).tolist()

            self._stats2["data"].extend(loaded_data["data"])
            self._stats2["refs"].extend(loaded_data["refs"])

        for idx, obj in enumerate(self._stats2["data"], 1):
            obj[0] = idx

        # return np.array(res, dtype=object)
        return None


class ClassSizes(BaseStats):
    """
    Columns:
        Class
        Object count
        Min height px
        Min height %
        Max height px
        Max height %
        Avg height px
        Avg height %
        Min width px
        Min width %
        Max width px
        Max width %
        Avg width px
        Avg width %
        Min area px
        Min area %
        Max area px
        Max area %
        Avg area px
        Avg area %
    """

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False):
        self._meta = project_meta
        self.force = force
        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]

        self._data = []

        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes.items()}

    def clean(self):
        self.__init__(
            self._meta,
            self.force,
        )

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            return
        lite_labels = []

        for figure in figures:
            lite_labels.append(
                LiteLabel(
                    obj_class_name=self._class_ids[figure.class_id],
                    geometry_type=figure.geometry_type,
                    geometry_to_bbox=figure.bbox,
                    geometry_area=figure.area,
                )
            )
        lite_ann = LiteAnnotation(labels=lite_labels, img_size=(image.height, image.width))

        self._data.append(lite_ann)

    def to_json2(self) -> Dict:
        return self.to_json()

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        lite_labels = [
            LiteLabel(
                obj_class_name=label.obj_class.name,
                geometry_type=type(label.geometry),
                geometry_to_bbox=label.geometry.to_bbox(),
                geometry_area=label.geometry.area,
            )
            for label in ann.labels
        ]
        lite_ann = LiteAnnotation(labels=lite_labels, img_size=ann.img_size)

        self._data.append(lite_ann)

    def to_json(self) -> Dict:
        # if not self._data:
        #     sly.logger.warning(
        #         "ClassSizes: No stats were added in update() method, the result will be None."
        #     )
        #     return

        stats = []

        class_heights_px = defaultdict(list)
        class_heights_pc = defaultdict(list)
        class_widths_px = defaultdict(list)
        class_widths_pc = defaultdict(list)
        class_areas_pc = defaultdict(list)
        class_object_counts = defaultdict(int)

        for ann in self._data:
            image_height, image_width = ann.img_size
            for label in ann.labels:
                # if type(label.geometry) not in [sly.Bitmap, sly.Rectangle, sly.Polygon]:
                if label.geometry_type not in [
                    sly.Bitmap.name(),
                    sly.Rectangle.name(),
                    sly.Polygon.name(),
                    sly.GraphNodes.name(),
                    sly.Point.name(),
                ]:
                    continue

                # class_object_counts[label.obj_class.name] += 1
                class_object_counts[label.obj_class_name] += 1

                obj_sizes = calculate_obj_sizes(label, image_height, image_width)

                class_heights_px[label.obj_class_name].append(obj_sizes["height_px"])
                class_heights_pc[label.obj_class_name].append(obj_sizes["height_pc"])
                class_widths_px[label.obj_class_name].append(obj_sizes["width_px"])
                class_widths_pc[label.obj_class_name].append(obj_sizes["width_pc"])
                class_areas_pc[label.obj_class_name].append(obj_sizes["area_pc"])

        for class_title in self._class_titles:
            object_count = class_object_counts.get(class_title, 0)

            cls_area = class_areas_pc[class_title]
            avg_area_pc = 0 if len(cls_area) == 0 else sum(cls_area) / len(cls_area)

            cls_hts_px = class_heights_px[class_title]
            avg_height_px = 0 if len(cls_hts_px) == 0 else sum(cls_hts_px) / len(cls_hts_px)

            cls_hts_pc = class_heights_pc[class_title]
            avg_height_pc = 0 if len(cls_hts_pc) == 0 else sum(cls_hts_pc) / len(cls_hts_pc)

            cls_wdt_px = class_widths_px[class_title]
            avg_width_px = 0 if len(cls_wdt_px) == 0 else sum(cls_wdt_px) / len(cls_wdt_px)

            cls_wdt_pc = class_widths_pc[class_title]
            avg_width_pc = 0 if len(cls_wdt_pc) == 0 else sum(cls_wdt_pc) / len(cls_wdt_pc)

            class_data = {
                "class_name": class_title,
                "object_count": object_count,
                "avg_area_pc": round(avg_area_pc, 2),
                "max_area_pc": max(cls_area, default=0),
                "min_area_pc": min(cls_area, default=0),
                "min_height_px": min(cls_hts_px, default=0),
                "min_height_pc": min(cls_hts_pc, default=0),
                "max_height_px": max(cls_hts_px, default=0),
                "max_height_pc": max(cls_hts_pc, default=0),
                "avg_height_px": round(avg_height_px, 2),
                "avg_height_pc": round(avg_height_pc, 2),
                "min_width_px": min(class_widths_px[class_title], default=0),
                "min_width_pc": min(class_widths_pc[class_title], default=0),
                "max_width_px": max(class_widths_px[class_title], default=0),
                "max_width_pc": max(class_widths_pc[class_title], default=0),
                "avg_width_px": round(avg_width_px, 2),
                "avg_width_pc": round(avg_width_pc, 2),
            }

            class_data = list(class_data.values())

            stats.append(class_data)

        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "desc"},
            "pageSize": 10,
        }

        res = {
            "columns": [
                "Class",
                "Object count",
                "Avg area",
                "Max area",
                "Min area",
                "Min height",
                "Min height",
                "Max height",
                "Max height",
                "Avg height",
                "Avg height",
                "Min width",
                "Min width",
                "Max width",
                "Max width",
                "Avg width",
                "Avg width",
            ],
            "columnsOptions": [
                {"type": "class"},
                {"maxValue": max([class_data[1] for class_data in stats])},
                {
                    "postfix": "%",
                    "tooltip": "Average object area in percents of all image.",
                },
                {
                    "postfix": "%",
                    "tooltip": "Maximum object area in percents of all image.",
                },
                {
                    "postfix": "%",
                    "tooltip": "Minimum object area in percents of all image.",
                },
                {"postfix": "px"},
                {
                    "postfix": "%",
                    "tooltip": "Minimum object height in percents of image height.",
                },
                {"postfix": "px"},
                {
                    "postfix": "%",
                    "tooltip": "Maximum object height in percents of image height.",
                },
                {"postfix": "px"},
                {
                    "postfix": "%",
                    "tooltip": "Average object height in percents of image height.",
                },
                {"postfix": "px"},
                {
                    "postfix": "%",
                    "tooltip": "Minimum object width in percents of image width.",
                },
                {"postfix": "px"},
                {
                    "postfix": "%",
                    "tooltip": "Maximum object width in percents of image width.",
                },
                {"postfix": "px"},
                {
                    "postfix": "%",
                    "tooltip": "Average object width in percents of image width.",
                },
            ],
            "data": stats,
            "options": options,
        }

        return res

    def to_numpy_raw(self) -> np.ndarray:
        return np.array(self._data, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir, *args, **kwargs) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        for file in files:
            loaded_data = np.load(file, allow_pickle=True)

            for image in loaded_data.tolist():
                labels, img_size = image
                lite_ann = LiteAnnotation(labels, img_size)
                self._data.append(lite_ann)

        return np.array(self._data, dtype=object)


class ClassesTreemap(BaseStats):
    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False):
        self._meta = project_meta
        self.force = force

        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]
        self._number_of_classes = len(self._class_titles)
        self._class_rgbs = [obj_class.color for obj_class in project_meta.obj_classes]
        self._class_colors = [rgb_to_hex(rgb) for rgb in self._class_rgbs]

        self._data = []

        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes.items()}

    def clean(self):
        self.__init__(
            self._meta,
            self.force,
        )

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            return
        lite_labels = []

        for figure in figures:
            lite_labels.append(
                LiteLabel(
                    obj_class_name=self._class_ids[figure.class_id],
                    geometry_type=figure.geometry_type,
                    geometry_to_bbox=figure.bbox,
                    geometry_area=figure.area,
                )
            )
        lite_ann = LiteAnnotation(labels=lite_labels, img_size=(image.height, image.width))

        self._data.append(lite_ann)

    def to_json2(self) -> Dict:
        return self.to_json()

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        lite_labels = [
            LiteLabel(
                obj_class_name=label.obj_class.name,
                geometry_type=type(label.geometry),
                geometry_to_bbox=label.geometry.to_bbox(),
                geometry_area=label.geometry.area,
            )
            for label in ann.labels
        ]
        lite_ann = LiteAnnotation(labels=lite_labels, img_size=ann.img_size)

        self._data.append(lite_ann)

    def to_json(self) -> Dict:
        if not self._data:
            sly.logger.warning(
                "ClassesTreemap: No stats were added in update() method, the result will be None."
            )
            return

        tooltip = "Average area of class objects on image is {y}%"
        colors = self._class_colors
        names = []
        values = []

        if self._number_of_classes < 2:
            return

        class_areas_pc = defaultdict(list)
        class_object_counts = defaultdict(int)

        for ann in self._data:
            image_height, image_width = ann.img_size
            for label in ann.labels:
                # if type(label.geometry) not in [sly.Bitmap, sly.Rectangle, sly.Polygon]:
                if label.geometry_type not in [
                    sly.Bitmap.geometry_name(),
                    sly.Rectangle.geometry_name(),
                    sly.Polygon.geometry_name(),
                ]:
                    continue

                class_object_counts[label.obj_class_name] += 1

                obj_sizes = calculate_obj_sizes(label, image_height, image_width)

                class_areas_pc[label.obj_class_name].append(obj_sizes["area_pc"])

        for class_title in self._class_titles:
            object_count = class_object_counts[class_title]

            if object_count < 1:
                continue

            names.append(class_title)
            values.append(
                round(
                    sum(class_areas_pc[class_title]) / len(class_areas_pc[class_title]),
                    2,
                )
            )

        tc = TreemapChart(
            title="Average area of class objects on image",
            colors=colors,
            tooltip=tooltip,
        )

        tc.set_series(names, values)

        res = tc.get_json_data()

        if 10 > self._number_of_classes >= 2:
            class_height = 40
        elif 20 > self._number_of_classes >= 10:
            class_height = 30
        else:
            class_height = 20

        max_widget_height = 800
        calculated_height = self._number_of_classes * class_height
        height = min(calculated_height, max_widget_height) + 150

        res["options"]["colors"] = colors
        res["options"]["chart"]["height"] = height

        return res

    def to_numpy_raw(self) -> np.ndarray:
        return np.array(self._data, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir, *args, **kwargs) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        # TODO handle when class has 0 images

        for file in files:
            loaded_data = np.load(file, allow_pickle=True)

            for image in loaded_data.tolist():
                labels, img_size = image
                lite_ann = LiteAnnotation(labels, img_size)
                self._data.append(lite_ann)

        return np.array(self._data, dtype=object)


def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def calculate_obj_sizes(label: sly.Label, image_height: int, image_width: int) -> Dict:
    image_area = image_height * image_width

    rect_geometry = label.geometry_to_bbox
    if rect_geometry is not None:
        height_px = rect_geometry.height
        width_px = rect_geometry.width
    else:
        height_px, width_px = 0, 0

    height_pc = round(height_px * 100.0 / image_height, 2)
    width_pc = round(width_px * 100.0 / image_width, 2)

    area_px = int(label.geometry_area)
    area_pc = round(area_px * 100.0 / image_area, 2)

    return {
        "height_px": height_px,
        "height_pc": height_pc,
        "width_px": width_px,
        "width_pc": width_pc,
        "area_pc": area_pc,
    }

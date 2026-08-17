import math
import os
import random
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

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
            # if figure.geometry_type not in [
            #     sly.Bitmap.name(),
            #     sly.Rectangle.name(),
            #     sly.Polygon.name(),
            #     sly.GraphNodes.name(),
            #     sly.Point.name(),
            # ]:
            #     continue

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
    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        def custom_key(path):
            # Split path and extract dataset ID and chunk ID
            parts = os.path.basename(path).split("_")
            return int(parts[2]), int(parts[1])

        # Sort paths by dataset ID and then by chunk ID
        sorted_files = sorted(files, key=custom_key)

        # One row per object in the project used to be merged and serialized in full, even though
        # update_freq -- computed above from the same budget -- says this table is meant to hold
        # at most MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF rows. Only the single-pass update()
        # path ever applied it, so the chunked path grew with the object count instead.
        # Reservoir sampling applies the budget while merging, so memory is bounded by the budget
        # rather than by the project, and the retained sample stays uniform over all chunks.
        # Seeded like every other sampling decision in this package, so runs are reproducible.
        limit = int(MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF)
        rng = random.Random(42)
        data = self._stats2["data"]
        refs = self._stats2["refs"]
        seen = 0

        for file in sorted_files:
            loaded_data = np.load(file, allow_pickle=True).tolist()

            for row, ref in zip(loaded_data["data"], loaded_data["refs"]):
                if len(data) < limit:
                    data.append(row)
                    refs.append(ref)
                else:
                    # Replace with probability limit/seen, the standard reservoir step.
                    pos = rng.randint(0, seen)
                    if pos < limit:
                        data[pos] = row
                        refs[pos] = ref
                seen += 1

            del loaded_data

        for idx, obj in enumerate(data, 1):
            obj[0] = idx

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
        self._acc = {}

        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes.items()}

    def clean(self):
        self.__init__(self._meta, self.force)

    _METRICS = ("height_px", "height_pc", "width_px", "width_pc", "area_pc")

    def _fold_annotation(self, ann) -> None:
        """Accumulate one annotation into the per-class running totals."""
        image_height, image_width = ann.img_size

        for label in ann.labels:
            acc = self._acc.get(label.obj_class_name)
            if acc is None:
                acc = self._acc[label.obj_class_name] = {
                    "count": 0,
                    **{metric: {"sum": 0.0, "min": None, "max": None} for metric in self._METRICS},
                }

            acc["count"] += 1
            obj_sizes = calculate_obj_sizes(label, image_height, image_width)

            for metric in self._METRICS:
                value = obj_sizes[metric]
                bucket = acc[metric]
                bucket["sum"] += value
                if bucket["min"] is None or value < bucket["min"]:
                    bucket["min"] = value
                if bucket["max"] is None or value > bucket["max"]:
                    bucket["max"] = value

    def _drain_data(self) -> None:
        """Fold whatever update() collected, so to_json() can read the accumulators alone."""
        while self._data:
            self._fold_annotation(self._data.pop())

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
        if len(self._class_titles) == 0:
            return

        stats = []

        self._drain_data()

        def agg(class_title: str, metric: str) -> Tuple[float, float, float]:
            """(avg, min, max) for one metric, matching the empty-class defaults of 0."""
            acc = self._acc.get(class_title)
            if acc is None or acc["count"] == 0:
                return 0, 0, 0
            bucket = acc[metric]
            return bucket["sum"] / acc["count"], bucket["min"], bucket["max"]

        for class_title in self._class_titles:
            acc = self._acc.get(class_title)
            object_count = 0 if acc is None else acc["count"]

            avg_area_pc, min_area_pc, max_area_pc = agg(class_title, "area_pc")
            avg_height_px, min_height_px, max_height_px = agg(class_title, "height_px")
            avg_height_pc, min_height_pc, max_height_pc = agg(class_title, "height_pc")
            avg_width_px, min_width_px, max_width_px = agg(class_title, "width_px")
            avg_width_pc, min_width_pc, max_width_pc = agg(class_title, "width_pc")

            class_data = {
                "class_name": class_title,
                "object_count": object_count,
                "avg_area_pc": round(avg_area_pc, 2),
                "max_area_pc": max_area_pc,
                "min_area_pc": min_area_pc,
                "min_height_px": min_height_px,
                "min_height_pc": min_height_pc,
                "max_height_px": max_height_px,
                "max_height_pc": max_height_pc,
                "avg_height_px": round(avg_height_px, 2),
                "avg_height_pc": round(avg_height_pc, 2),
                "min_width_px": min_width_px,
                "min_width_pc": min_width_pc,
                "max_width_px": max_width_px,
                "max_width_pc": max_width_pc,
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
    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        # Folded per chunk instead of collected: every figure in the project used to be held as
        # a LiteLabel until to_json() ran, and this chart only reports count/min/max/mean per
        # class -- all of which accumulate exactly. Memory is now O(classes), not O(objects),
        # and the numbers are unchanged.
        for file in files:
            loaded_data = np.load(file, allow_pickle=True)

            for image in loaded_data.tolist():
                labels, img_size = image
                self._fold_annotation(LiteAnnotation(labels, img_size))

            del loaded_data

        return None


class ClassesTreemap(BaseStats):
    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False):
        self._meta = project_meta
        self.force = force

        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]
        self._number_of_classes = len(self._class_titles)
        self._class_rgbs = [obj_class.color for obj_class in project_meta.obj_classes]
        self._class_colors = [rgb_to_hex(rgb) for rgb in self._class_rgbs]

        self._data = []
        self._acc = {}

        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes.items()}

    def clean(self):
        self.__init__(self._meta, self.force)

    _AREA_GEOMETRIES = (
        sly.Bitmap.geometry_name(),
        sly.Rectangle.geometry_name(),
        sly.Polygon.geometry_name(),
    )

    def _fold_annotation(self, ann) -> None:
        """Accumulate one annotation into the per-class area totals."""
        image_height, image_width = ann.img_size

        for label in ann.labels:
            if label.geometry_type not in self._AREA_GEOMETRIES:
                continue

            acc = self._acc.get(label.obj_class_name)
            if acc is None:
                acc = self._acc[label.obj_class_name] = {"count": 0, "area_sum": 0.0}

            acc["count"] += 1
            acc["area_sum"] += calculate_obj_sizes(label, image_height, image_width)["area_pc"]

    def _drain_data(self) -> None:
        """Fold whatever update() collected, so to_json() can read the accumulators alone."""
        while self._data:
            self._fold_annotation(self._data.pop())

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
        self._drain_data()

        if not self._acc:
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

        for class_title in self._class_titles:
            acc = self._acc.get(class_title)

            if acc is None or acc["count"] < 1:
                continue

            names.append(class_title)
            values.append(round(acc["area_sum"] / acc["count"], 2))

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
    def sew_chunks(self, chunks_dir) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        # TODO handle when class has 0 images

        # Folded per chunk rather than collected -- see ClassSizes.sew_chunks. This chart only
        # reports one average per class, so nothing needs the per-object rows to survive.
        for file in files:
            loaded_data = np.load(file, allow_pickle=True)

            for image in loaded_data.tolist():
                labels, img_size = image
                self._fold_annotation(LiteAnnotation(labels, img_size))

            del loaded_data

        return None


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

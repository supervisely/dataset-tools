from collections import defaultdict, namedtuple
from typing import Dict, List, Optional
from itertools import groupby
import supervisely as sly
import numpy as np
from supervisely.app.widgets import HeatmapChart
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from dataset_tools.image.stats.basestats import BaseStats

from supervisely.imaging.color import random_rgb

MAX_NUMBER_OF_COLUMNS = 100
LiteLabel = namedtuple("LiteLabel", ["obj_class_name"])
LiteAnnotation = namedtuple("LiteAnnotation", ["labels"])
REFERENCES_LIMIT = 1000


def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class ObjectsDistribution(BaseStats):
    """
    Columns:
        Class
        1 object on image (if object exists)
        2 objects on image (if objects exist)
        3 objects on image (if objects exist)
        etc.
    """

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        force: bool = False,
    ):
        self._meta = project_meta
        self.force = force

        #  !old
        self._obj_classes = project_meta.obj_classes
        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]

        self._images = []
        self._anns = []

        # new
        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes}
        self._class_ids = dict(sorted(self._class_ids.items(), key=lambda item: item[1]))

        self._distribution_dict = {class_id: {0: set()} for class_id in self._class_ids}
        self._max_count = 0
        self._classes_hex = {item.sly_id: rgb_to_hex(item.color) for item in self._meta.obj_classes}

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            for class_id in self._class_ids:
                self._distribution_dict[class_id][0].add(image.id)
            return

        nonzero = []
        sorted_figures = sorted(figures, key=lambda x: x.class_id)

        for class_id, group in groupby(sorted_figures, key=lambda x: x.class_id):
            count = len(list(group))
            num_img = self._distribution_dict[class_id].get(count)
            if num_img is None:
                self._distribution_dict[class_id][count] = set()
            self._distribution_dict[class_id][count].add(image.id)
            nonzero.append(class_id)
            self._max_count = count if count > self._max_count else self._max_count

        for class_id in self._class_ids:
            if class_id not in nonzero:
                self._distribution_dict[class_id][0].add(image.id)

    def to_json2(self) -> Dict:
        series, colors = [], []
        references = defaultdict(dict)
        axis = [i for i in range(self._max_count + 1)]
        for class_id, class_name in self._class_ids.items():
            class_ditrib = self._distribution_dict[class_id]
            reference = {x: [] for x in axis}

            values = [0 for _ in range(self._max_count + 1)]
            for objects_count, images_set in class_ditrib.items():
                values[objects_count] = len(images_set)

                seized_refs = self._seize_list_to_fixed_size(list(images_set), REFERENCES_LIMIT)
                reference[objects_count] = seized_refs
                references.setdefault(class_name, {}).update(reference)

            row = {
                "name": class_name,
                "y": values,
                "x": axis,
            }

            series.append(row)
            colors.append(self._classes_hex[class_id])

        hmp = HeatmapChart(
            title="Objects on images - distribution for every class",
            xaxis_title="Number of objects on image",
            color_range="row",
            tooltip="Click to preview {y} images with {x} objects of class {series_name}",
        )
        hmp.add_series_batch(series)

        number_of_rows = len(series)
        max_widget_height = 10000
        if number_of_rows < 5:
            row_height = 70
        elif number_of_rows < 20:
            row_height = 50
        else:
            row_height = 30

        res = hmp.get_json_data()
        number_of_columns = len(axis)
        calculated_height = number_of_rows * row_height
        height = min(calculated_height, max_widget_height) + 150
        res["referencesCell"] = references
        res["options"]["chart"]["height"] = height
        res["options"]["colors"] = colors

        # Disabling labels and ticks for x-axis if there are too many columns.
        if MAX_NUMBER_OF_COLUMNS > number_of_columns > 40:
            res["options"]["xaxis"]["labels"] = {"show": False}
            res["options"]["xaxis"]["axisTicks"] = {"show": False}
            res["options"]["dataLabels"] = {"enabled": False}
        elif number_of_columns >= MAX_NUMBER_OF_COLUMNS:
            return

        return res

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        self._images.append(image.id)

        lite_labels = [LiteLabel(obj_class_name=label.obj_class.name) for label in ann.labels]
        lite_ann = LiteAnnotation(labels=lite_labels)

        self._anns.append(lite_ann)

    def to_json(self) -> Dict:
        if len(self._images) == 0:
            sly.logger.warning(
                "ObjectDistribution: No stats were added in update() method, the result will be None."
            )
            return

        self._stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "image_ids": []}))
        counters = defaultdict(lambda: {"count": 0, "image_ids": []})

        for image_id, ann in zip(self._images, self._anns):
            # image_id = image.id
            counters = defaultdict(lambda: {"count": 0, "image_ids": []})

            for class_title in self._class_titles:
                if class_title not in [label.obj_class_name for label in ann.labels]:
                    counters[class_title]["image_ids"].append(image_id)

            for label in ann.labels:
                class_title = label.obj_class_name
                counters[class_title]["count"] += 1
                counters[class_title]["image_ids"].append(image_id)

            for class_title in self._class_titles:
                count = counters[class_title]["count"]
                image_ids = counters[class_title]["image_ids"]
                self._stats[class_title][count]["image_ids"].extend(list(set(image_ids)))
                self._stats[class_title][count]["count"] += 1

        max_column = max([max(class_data.keys()) for class_data in self._stats.values()])
        columns = [i for i in range(max_column + 1)]

        series = list()
        colors = list()
        for class_title, class_data in self._stats.items():
            row = {
                "name": class_title,
                "y": [class_data[column]["count"] for column in columns],
                "x": columns,
            }

            series.append(row)
            for obj_class in self._obj_classes:
                if obj_class.name == class_title:
                    color = obj_class.color
                    break
            colors.append(rgb_to_hex(color))

        references = defaultdict(dict)

        for column in columns:
            for class_title, class_data in self._stats.items():
                image_ids = class_data[column]["image_ids"]
                reference = {
                    column: image_ids,
                }
                if references[class_title]:
                    references[class_title].update(reference)
                else:
                    references[class_title] = reference

        hmp = HeatmapChart(
            # title="Objects on images - distribution for every class",
            title="",
            xaxis_title="Number of objects on image",
            color_range="row",
            tooltip="Click to preview {y} images with {x} objects of class {series_name}",
        )
        hmp.add_series_batch(series)

        number_of_rows = len(series)
        max_widget_height = 10000
        if number_of_rows < 5:
            row_height = 70
        elif number_of_rows < 20:
            row_height = 50
        else:
            row_height = 30

        res = hmp.get_json_data()
        number_of_columns = len(columns)
        calculated_height = number_of_rows * row_height
        height = min(calculated_height, max_widget_height) + 150
        res["referencesCell"] = references
        res["options"]["chart"]["height"] = height
        res["options"]["colors"] = colors

        # Disabling labels and ticks for x-axis if there are too many columns.
        if MAX_NUMBER_OF_COLUMNS > number_of_columns > 40:
            res["options"]["xaxis"]["labels"] = {"show": False}
            res["options"]["xaxis"]["axisTicks"] = {"show": False}
            res["options"]["dataLabels"] = {"enabled": False}
        elif number_of_columns >= MAX_NUMBER_OF_COLUMNS:
            return

        return res

    def to_numpy_raw(self):
        # images = [
        #     [im_id] + [lbl.obj_class_name for lbl in ann.labels]
        #     for im_id, ann in zip(self._images, self._anns)
        # ]

        return np.array(self._distribution_dict, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str, updated_classes: dict) -> np.ndarray:
        # if len(updated_classes) > 0:
        #     self._class_ids.update(updated_classes)

        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                loaded_classes = set([class_id for class_id in loaded_data])
                true_classes = set(self._class_ids)

                added = true_classes - loaded_classes
                for class_id in list(added):
                    if loaded_data.get(class_id) is None:
                        loaded_data[class_id] = {0: set()}
                    for other_class in loaded_data:
                        for images_set in loaded_data[other_class].values():
                            loaded_data[class_id][0].update(images_set)

                removed = loaded_classes - true_classes
                for class_id in list(removed):
                    loaded_data.pop(class_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for class_id in self._class_ids:
                    for objects_count, images_set in loaded_data[class_id].items():
                        try:
                            self._distribution_dict[class_id][objects_count].update(images_set)
                        except KeyError:
                            self._distribution_dict[class_id][objects_count] = images_set
                        self._max_count = max(self._max_count, objects_count)

        return None

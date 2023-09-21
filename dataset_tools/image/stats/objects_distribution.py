from collections import defaultdict, namedtuple
from typing import Dict, List

import supervisely as sly
from supervisely.app.widgets import HeatmapChart

from dataset_tools.image.stats.basestats import BaseStats

MAX_NUMBER_OF_COLUMNS = 100
LiteLabel = namedtuple("LiteLabel", ["obj_class_name"])
LiteAnnotation = namedtuple("LiteAnnotation", ["labels"])


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
        self.force = force

        self.project_meta = project_meta
        self._counters = defaultdict(lambda: {"count": 0, "image_ids": []})
        self._obj_classes = project_meta.obj_classes
        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]

        self._images = []
        self._anns = []

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        self._images.append(image)

        lite_labels = [LiteLabel(obj_class_name=label.obj_class.name) for label in ann.labels]
        lite_ann = LiteAnnotation(labels=lite_labels)

        self._anns.append(lite_ann)

    def to_json(self) -> Dict:
        if not self._images:
            sly.logger.warning("No stats were added in update() method, the result will be None.")
            return

        self._stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "image_ids": []}))
        counters = defaultdict(lambda: {"count": 0, "image_ids": []})

        for image, ann in zip(self._images, self._anns):
            image_id = image.id
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


def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

from collections import defaultdict

from typing import Dict

import supervisely as sly
from supervisely.app.widgets import HeatmapChart

from dataset_tools.image.stats.basestats import BaseStats


class ObjectsDistribution(BaseStats):
    """
    Columns:
        Class
        1 object on image (if object exists)
        2 objects on image (if objects exist)
        3 objects on image (if objects exist)
        etc.
    """

    def __init__(self, project_meta: sly.ProjectMeta, force:bool = False):
        self.force = force

        self.project_meta = project_meta
        self._counters = defaultdict(lambda: {"count": 0, "image_ids": []})
        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]
        self._data = []

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        self._data.append((image, ann))

    def to_json(self) -> Dict:
        self._stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "image_ids": []}))
        counters = defaultdict(lambda: {"count": 0, "image_ids": []})

        for image, ann in self._data:
            image_id = image.id
            counters = defaultdict(lambda: {"count": 0, "image_ids": []})

            for label in ann.labels:
                class_title = label.obj_class.name
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
        for class_title, class_data in self._stats.items():
            row = {
                "name": class_title,
                "y": [class_data[column]["count"] for column in columns],
                "x": columns,
            }

            series.append(row)

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

        res = hmp.get_json_data()
        res["referencesCell"] = references

        return res

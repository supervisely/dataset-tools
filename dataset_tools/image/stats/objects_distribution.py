from collections import defaultdict
from dataset_tools.image.stats.basestats import BaseStats

import pandas as pd
import supervisely as sly


class ObjectsDistribution(BaseStats):
    """
    Columns:
        class name
        1 object on image (if object exists)
        2 objects on image (if objects exist)
        3 objects on image (if objects exist)
        etc.
    """

    def __init__(self, project_meta: sly.ProjectMeta):
        self.project_meta = project_meta
        self._counters = defaultdict(lambda: {"count": 0, "image_ids": []})
        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]
        self._data = []

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        self._data.append((image, ann))

    def to_json(self):
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

        columns = set()
        for class_title, class_data in self._stats.items():
            columns.update(class_data.keys())

        columns = sorted(list(columns))

        data = list()
        for class_title, class_data in self._stats.items():
            row = [class_title]
            for column in columns:
                count = class_data[column]["count"]
                row.append(count)

            data.append(row)

        references = list()

        for column in columns:
            for class_title, class_data in self._stats.items():
                image_ids = class_data[column]["image_ids"]
                reference = {
                    column: image_ids,
                }

                references.append(reference)

        options = {"fixColumns": 1}

        res = {
            "columns": ["class name"] + columns,
            "data": data,
            "referencesRow": references,
            "options": options,
        }

        return res

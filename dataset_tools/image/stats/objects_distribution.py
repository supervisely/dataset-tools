from collections import defaultdict

import pandas as pd
import supervisely as sly


class ObjectsDistribution:
    def __init__(self, project_meta: sly.ProjectMeta):
        self.project_meta = project_meta
        self._stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "image_ids": []}))
        self._counters = defaultdict(lambda: {"count": 0, "image_ids": []})
        self._class_titles = [obj_class.name for obj_class in project_meta.obj_classes]

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        for label in ann.labels:
            class_title = label.obj_class.name
            self._counters[class_title]["count"] += 1
            self._counters[class_title]["image_ids"].append(image.id)

        for class_title in self._class_titles:
            count = self._counters[class_title]["count"]
            image_ids = self._counters[class_title]["image_ids"]
            self._stats[class_title][count]["image_ids"].extend(list(set(image_ids)))
            self._stats[class_title][count]["count"] += 1

    def to_json(self):
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
            "columns": columns,
            "data": data,
            "referencesRow": references,
            "options": options,
        }

        return res

    def to_pandas(self):
        json = self.to_json()
        table = pd.DataFrame(json["data"], columns=["class names"] + json["columns"])
        return table

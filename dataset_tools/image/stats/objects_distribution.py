from collections import defaultdict

import pandas as pd


class ObjectsDistribution:
    def __init__(self, project_data):
        self.project_meta = project_data.project_meta
        self.datasets = project_data.datasets
        self.class_titles = [obj_class.name for obj_class in project_data.project_meta.obj_classes]

    def update(self):
        self.classes = defaultdict(lambda: defaultdict(lambda: {"count": 0, "image_ids": []}))
        counters = defaultdict(lambda: {"count": 0, "image_ids": []})

        for dataset in self.datasets:
            anns = dataset.anns
            image_ids = [image_info.id for image_info in dataset.image_infos]

            for ann, image_id in zip(anns, image_ids):
                counters = defaultdict(lambda: {"count": 0, "image_ids": []})

                for label in ann.labels:
                    class_title = label.obj_class.name
                    counters[class_title]["count"] += 1
                    counters[class_title]["image_ids"].append(image_id)

                for class_title in self.class_titles:
                    count = counters[class_title]["count"]
                    image_ids = counters[class_title]["image_ids"]
                    self.classes[class_title][count]["image_ids"].extend(list(set(image_ids)))
                    self.classes[class_title][count]["count"] += 1

    def to_json(self):
        columns = set()
        for class_title, class_data in self.classes.items():
            columns.update(class_data.keys())

        columns = sorted(list(columns))

        data = list()
        for class_title, class_data in self.classes.items():
            row = [class_title]
            for column in columns:
                count = class_data[column]["count"]
                row.append(count)

            data.append(row)

        references = list()

        for column in columns:
            for class_title, class_data in self.classes.items():
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
        table = pd.DataFrame(json["data"], columns=json["columns"])
        return table

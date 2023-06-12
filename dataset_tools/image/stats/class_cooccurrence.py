from collections import defaultdict
from typing import Dict

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

# from supervisely.app.widgets import ConfusionMatrix


class ClassCooccurrence(BaseStats):
    """
    Columns:
        Class
        class 1
        class 2
        etc.
    """

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force

        self._name_to_index = {}

        for idx, obj_class in enumerate(self._meta.obj_classes):
            self._name_to_index[obj_class.name] = idx

        self._class_names = [cls.name for cls in project_meta.obj_classes]
        self._references = defaultdict(lambda: defaultdict(list))

        num_classes = len(self._class_names)
        self.co_occurrence_matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        classes = set()
        for label in ann.labels:
            classes.add(label.obj_class.name)

        for class_ in classes:
            idx = self._name_to_index[class_]
            self.co_occurrence_matrix[idx][idx] += 1
            self._references[idx][idx].append(image.id)

        classes = list(classes)
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class_i = classes[i]
                class_j = classes[j]
                idx_i = self._name_to_index[class_i]
                idx_j = self._name_to_index[class_j]
                self.co_occurrence_matrix[idx_i][idx_j] += 1
                self.co_occurrence_matrix[idx_j][idx_i] += 1

                self._references[idx_i][idx_j].append(image.id)
                self._references[idx_j][idx_i].append(image.id)

    def to_json(self) -> Dict:
        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
        }
        colomns_options = [None] * (len(self._class_names) + 1)
        colomns_options[0] = {"type": "class"}  # not used in Web

        for idx in range(1, len(colomns_options)):
            colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

        data = [
            [value] + sublist
            for value, sublist in zip(self._class_names, self.co_occurrence_matrix.tolist())
        ]

        res = {
            "columns": ["Class"] + self._class_names,
            "data": data,
            "referencesCell": self._references,  # row - col
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    # def get_widget(self) -> ConfusionMatrix:
    #     df = pd.DataFrame(data=self.co_occurrence_matrix.tolist(), columns=self._class_names)
    #     confusion_matrix = ConfusionMatrix()
    #     confusion_matrix.read_pandas(df)
    #     return confusion_matrix

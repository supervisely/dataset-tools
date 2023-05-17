import itertools
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import dataframe_image as dfi
import numpy as np
import pandas as pd
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats
from supervisely.app.widgets import ConfusionMatrix


class ClassCooccurrence(BaseStats):
    """
    Columns:
        class name
        class 1
        class 2
        etc.
    """

    def __init__(self, project_meta: sly.ProjectMeta) -> None:
        self._meta = project_meta
        self._stats = {}

        self._class_names = [cls.name for cls in project_meta.obj_classes]
        self._references = defaultdict(lambda: defaultdict(list))

        num_classes = len(self._class_names)
        self.co_occurrence_matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        classes = set()
        for label in ann.labels:
            classes.add(label.obj_class.name)

        classes = list(classes)
        for class_ in classes:
            idx = self._class_names.index(class_)
            self.co_occurrence_matrix[idx][idx] += 1
            self._references[idx][idx].append(image.id)

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class_i = classes[i]
                class_j = classes[j]
                idx_i = list(self._class_names).index(class_i)
                idx_j = list(self._class_names).index(class_j)
                self.co_occurrence_matrix[idx_i][idx_j] += 1
                self.co_occurrence_matrix[idx_j][idx_i] += 1

                self._references[idx_i][idx_j].append(image.id)
                self._references[idx_j][idx_i].append(image.id)

    def to_json(self):
        options = {"fixColumns": 1}
        colomns_options = [None] * (len(self._class_names) + 1)
        colomns_options[0] = {"type": "class"}

        for idx in range(1, len(colomns_options)):
            colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

        data = [
            [value] + sublist
            for value, sublist in zip(self._class_names, self.co_occurrence_matrix.tolist())
        ]

        res = {
            "columns": ["class name"] + self._class_names,
            "data": data,
            "referencesRow": self._references,
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def get_widget(self) -> ConfusionMatrix:
        df = pd.DataFrame(data=self.co_occurrence_matrix.tolist(), columns=self._class_names)
        confusion_matrix = ConfusionMatrix()
        confusion_matrix.read_pandas(df)
        return confusion_matrix

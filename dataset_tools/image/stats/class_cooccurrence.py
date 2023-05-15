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


class ClassCooccurence:
    """
    Important fields of modified stats dict:
        "class_names": [],
        "counters": [],
        "pd_data": [],
    """

    def __init__(self, project_meta: sly.ProjectMeta) -> None:
        self._meta = project_meta
        self._stats = {}

        self.class_names = [cls.name for cls in project_meta.obj_classes]

        self._stats["counters"] = defaultdict(lambda: defaultdict(list))

        num_classes = len(self.class_names)
        self.co_occurrence_matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        classes = set()
        for label in ann.labels:
            classes.add(label.obj_class.name)

        classes = list(classes)
        for class_ in classes:
            idx = self.class_names.index(class_)
            self.co_occurrence_matrix[idx][idx] += 1
            self._stats["counters"][class_][class_].append(image.id)

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class_i = classes[i]
                class_j = classes[j]
                idx_i = list(self.class_names).index(class_i)
                idx_j = list(self.class_names).index(class_j)
                self.co_occurrence_matrix[idx_i][idx_j] += 1
                self.co_occurrence_matrix[idx_j][idx_i] += 1

                self._stats["counters"][class_i][class_j].append(image.id)
                self._stats["counters"][class_j][class_i].append(image.id)

    def to_json(self):
        options = {"fixColumns": 1}

        colomns_options = [None] * (len(self.class_names) + 1)
        colomns_options[0] = {"type": "class"}

        for idx in range(1, len(colomns_options)):
            colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

        res = {
            "columns": self.class_names,
            "data": self.co_occurrence_matrix.tolist(),
            "referencesRow": self._stats["counters"],
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_pandas(self) -> pd.DataFrame:
        json = self.to_json()
        table = pd.DataFrame(data=json["data"], columns=json["columns"])
        return table

    def to_image(self, path):
        table = self.to_pandas()
        table.dfi.export(path)

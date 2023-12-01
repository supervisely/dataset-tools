from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import json, os

import supervisely as sly
from dataset_tools.image.stats.basestats import BaseStats

REFERENCES_LIMIT = 1000


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

        self._num_classes = len(self._class_names)
        self.co_occurrence_matrix = np.zeros(
            (self._num_classes, self._num_classes), dtype=int
        )

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

                if len(self._references[idx_i][idx_j]) <= REFERENCES_LIMIT:
                    self._references[idx_i][idx_j].append(image.id)
                if len(self._references[idx_j][idx_i]) <= REFERENCES_LIMIT:
                    self._references[idx_j][idx_i].append(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json(self) -> Optional[Dict]:
        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
        }
        colomns_options = [None] * (len(self._class_names) + 1)
        colomns_options[0] = {"type": "class"}  # not used in Web

        for idx in range(1, len(colomns_options)):
            colomns_options[idx] = {
                "maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))
            }

        data = [
            [value] + sublist
            for value, sublist in zip(
                self._class_names, self.co_occurrence_matrix.tolist()
            )
        ]

        res = {
            "columns": ["Class"] + self._class_names,
            "data": data,
            "referencesCell": self._references,  # row - col
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        matrix = np.array(self.co_occurrence_matrix, dtype="int32")

        n = self._num_classes
        ref_list = [[[] for _ in range(3)] for _ in range(3)]
        for i in range(n):
            for j in range(n):
                ref_list[i][j] = self._references[i][j]

        references = np.array(ref_list, dtype=object)

        return np.stack(
            [
                matrix,
                references,
            ],
            axis=0,
        )

    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        res = None
        is_zero_area = None
        references = None

        def merge_elements(a, b):
            if a is None:
                return b
            elif b is None:
                return a
            else:
                return [
                    elem1 + elem2
                    if elem1 is not None and elem2 is not None
                    else elem1 or elem2
                    for elem1, elem2 in zip(a, b)
                ]

        for file in files:
            loaded_data = np.load(file, allow_pickle=True)
            stat_data, ref_data = loaded_data[0], loaded_data[1]

            if res is None:
                res = np.zeros(stat_data.shape, dtype="int32")
            res = np.add(stat_data, res)

            if references is None:
                references = np.empty_like(ref_data).tolist()

            references = [
                merge_elements(sublist1, sublist2)
                for sublist1, sublist2 in zip(references, ref_data.tolist())
            ]

        self.co_occurrence_matrix = res
        for i, sublist in enumerate(references):
            for j, inner_list in enumerate(sublist):
                self._references[i][j] = inner_list

        return res

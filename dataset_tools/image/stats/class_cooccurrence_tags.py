import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

REFERENCES_LIMIT = 1000


class ClassCooccurrenceTags(BaseStats):
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
        self._sly_id_to_name = {}

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            self._name_to_index[tag_meta.name] = idx
            self._sly_id_to_name[tag_meta.sly_id] = tag_meta.name

        self._tag_names = [cls.name for cls in project_meta.tag_metas]
        self._references = defaultdict(lambda: defaultdict(list))

        self._num_tags = len(self._tag_names)
        self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        if self._num_tags == 1:
            return

        image_tags = set()
        for tag_info in image.tags:
            image_tags.add(self._sly_id_to_name[tag_info["tagId"]])

        for image_tag_ in image_tags:
            idx = self._name_to_index[image_tag_]
            self.co_occurrence_matrix[idx][idx] += 1
            self._references[idx][idx].append(image.id)

        image_tags = list(image_tags)
        for i in range(len(image_tags)):
            for j in range(i + 1, len(image_tags)):
                tag_i = image_tags[i]
                tag_j = image_tags[j]
                idx_i = self._name_to_index[tag_i]
                idx_j = self._name_to_index[tag_j]
                self.co_occurrence_matrix[idx_i][idx_j] += 1
                self.co_occurrence_matrix[idx_j][idx_i] += 1

                if len(self._references[idx_i][idx_j]) <= REFERENCES_LIMIT:
                    self._references[idx_i][idx_j].append(image.id)
                if len(self._references[idx_j][idx_i]) <= REFERENCES_LIMIT:
                    self._references[idx_j][idx_i].append(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json(self) -> Optional[Dict]:
        if self._num_tags <= 1:
            return None
        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
        }
        colomns_options = [None] * (len(self._tag_names) + 1)
        colomns_options[0] = {"type": "class"}  # not used in Web

        for idx in range(1, len(colomns_options)):
            colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

        data = [
            [value] + sublist
            for value, sublist in zip(self._tag_names, self.co_occurrence_matrix.tolist())
        ]

        res = {
            "columns": ["Tag"] + self._tag_names,
            "data": data,
            "referencesCell": self._references,  # row - col
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        if self._num_tags <= 1:
            return
        #  if unlabeled
        if np.sum(self.co_occurrence_matrix) == 0:
            return
        matrix = np.array(self.co_occurrence_matrix, dtype="int32")

        n = self._num_tags
        ref_list = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ref_list[i][j] = set(self._references[i][j])

        references = np.array(ref_list, dtype=object)

        return np.stack(
            [
                matrix,
                references,
            ],
            axis=0,
        )

    def sew_chunks(self, chunks_dir: str, updated_tags: List[str] = []) -> np.ndarray:
        if self._num_tags <= 1:
            return
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
                    (
                        elem1 + list(elem2)
                        if elem1 is not None and elem2 is not None
                        else elem1 or list(elem2)
                    )
                    for elem1, elem2 in zip(a, b)
                ]

        def update_shape(
            array: np.ndarray, updated_tags, insert_val=0
        ) -> Tuple[np.ndarray, np.ndarray]:
            if len(updated_tags) > 0:
                indices = list(sorted([self._tag_names.index(cls) for cls in updated_tags]))
                sdata = array[0].copy()
                rdata = array[1].copy().tolist()

                for ind in indices:
                    for axis in range(2):
                        sdata = np.apply_along_axis(
                            lambda line: np.insert(line, [ind], [insert_val]),
                            axis=axis,
                            arr=sdata,
                        )
                    empty_line = [[] for _ in range(len(rdata))]
                    rdata.insert(ind, empty_line)
                    rdata = [sublist[:ind] + [[]] + sublist[ind:] for sublist in rdata]

                return sdata, np.array(rdata, dtype=object)
            return array[0], array[1]

        for file in files:
            loaded_data = np.load(file, allow_pickle=True)
            if np.any(loaded_data == None):
                continue

            stat_data, ref_data = loaded_data[0], loaded_data[1]
            if loaded_data.shape[1] != self._num_tags:
                stat_data, ref_data = update_shape(loaded_data, updated_tags)

            if res is None:
                res = np.zeros((self._num_tags, self._num_tags), dtype="int32")
            res = np.add(stat_data, res)

            if references is None:
                references = np.empty_like(res).tolist()

            references = [
                merge_elements(sublist1, sublist2)
                for sublist1, sublist2 in zip(references, ref_data.tolist())
            ]

            np.save(file, np.stack([stat_data, ref_data]))

        self.co_occurrence_matrix = res
        for i, sublist in enumerate(references):
            for j, inner_list in enumerate(sublist):
                self._references[i][j] = list(inner_list)

        return res

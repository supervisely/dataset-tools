import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

REFERENCES_LIMIT = 1000
IMAGES_TAGS = ["imagesOnly", "all"]


class CooccurrenceImageTags(BaseStats):
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

        self._tag_name_to_type = {}

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            if tag_meta.applicable_to in IMAGES_TAGS:
                self._name_to_index[tag_meta.name] = idx
                self._sly_id_to_name[tag_meta.sly_id] = tag_meta.name
                self._tag_name_to_type[tag_meta.name] = tag_meta.value_type

        self._tag_names = [
            cls.name for cls in project_meta.tag_metas if cls.applicable_to in IMAGES_TAGS
        ]
        self._references = defaultdict(lambda: defaultdict(list))

        self._num_tags = len(self._tag_names)
        self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        if self._num_tags < 1:
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

        for idx in range(len(colomns_options) - 1):
            colomns_options[idx + 1] = {
                "subtitle": self._tag_name_to_type[self._tag_names[idx]],
            }
            # colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

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

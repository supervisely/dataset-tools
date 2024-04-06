import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

REFERENCES_LIMIT = 1000
IMAGES_TAGS = ["imagesOnly", "all"]
OBJECTS_TAGS = ["objectsOnly", "all"]
ONEOF_STRING = "oneof_string"


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

        self._images_tags = []
        self._tag_names = []

        for tag_meta in self._meta.tag_metas:
            if tag_meta.applicable_to in IMAGES_TAGS:
                self._images_tags.append(tag_meta)

        for idx, im_tag_meta in enumerate(self._images_tags):
            self._name_to_index[im_tag_meta.name] = idx
            self._sly_id_to_name[im_tag_meta.sly_id] = im_tag_meta.name
            self._tag_name_to_type[im_tag_meta.name] = im_tag_meta.value_type
            self._tag_names.append(im_tag_meta.name)

        self._references = defaultdict(lambda: defaultdict(list))

        self._num_tags = len(self._tag_names)
        self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

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


class CooccurrenceObjectTags(BaseStats):
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
        self._tag_names = []
        self._tag_name_to_type = {}

        self._object_tags = []

        for tag_meta in self._meta.tag_metas:
            if tag_meta.applicable_to in OBJECTS_TAGS:
                self._object_tags.append(tag_meta)

        for idx, obj_tag_meta in enumerate(self._object_tags):
            self._name_to_index[obj_tag_meta.name] = idx
            self._tag_name_to_type[obj_tag_meta.name] = obj_tag_meta.value_type
            self._tag_names.append(obj_tag_meta.name)

        self._references = defaultdict(lambda: defaultdict(list))

        self._num_tags = len(self._tag_names)
        self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

        for label in ann.labels:
            label_tags = set()

            for tag in label.tags:
                if tag.name in self._tag_names:
                    label_tag_ = tag.name
                    label_tags.add(tag.name)

            for label_tag_ in label_tags:
                idx = self._name_to_index[label_tag_]
                self.co_occurrence_matrix[idx][idx] += 1
                if image.id not in self._references[idx][idx]:
                    self._references[idx][idx].append(image.id)

            label_tags = list(label_tags)
            for i in range(len(label_tags)):
                for j in range(i + 1, len(label_tags)):
                    tag_i = label_tags[i]
                    tag_j = label_tags[j]
                    idx_i = self._name_to_index[tag_i]
                    idx_j = self._name_to_index[tag_j]
                    self.co_occurrence_matrix[idx_i][idx_j] += 1
                    self.co_occurrence_matrix[idx_j][idx_i] += 1

                    if len(self._references[idx_i][idx_j]) <= REFERENCES_LIMIT:
                        if image.id not in self._references[idx_i][idx_j]:
                            self._references[idx_i][idx_j].append(image.id)
                    if len(self._references[idx_j][idx_i]) <= REFERENCES_LIMIT:
                        if image.id not in self._references[idx_j][idx_i]:
                            self._references[idx_j][idx_i].append(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json(self) -> Optional[Dict]:
        if self._num_tags < 1:
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


class CooccurrenceOneOfStringTags(BaseStats):
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
        self._name_to_data = {}
        self._sly_id_to_name = {}

        self._tag_name_to_type = {}
        self._tag_names = []

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            if tag_meta.value_type == ONEOF_STRING:
                self._tag_names.append(tag_meta.name)
                curr_val_to_index = {}
                for idx, possible_val in enumerate(tag_meta.possible_values):
                    curr_val_to_index[possible_val] = idx
                    curr_references = defaultdict(list)
                    curr_co_occurrence_matrix = np.zeros((len(tag_meta.possible_values)), dtype=int)

                self._name_to_data[tag_meta.name] = (
                    curr_val_to_index,
                    curr_references,
                    curr_co_occurrence_matrix,
                )

                self._sly_id_to_name[tag_meta.sly_id] = tag_meta.name
        self._num_tags = len(list(self._name_to_data.keys()))

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

        for tag_info in image.tags:
            tag_name = self._sly_id_to_name.get(tag_info["tagId"])
            if tag_name in self._tag_names:
                tag_val = tag_info["value"]
                curr_tag_data = self._name_to_data[tag_name]
                idx = curr_tag_data[0][tag_val]
                curr_tag_data[2][idx] += 1
                if image.id not in curr_tag_data[1][idx]:
                    curr_tag_data[1][idx].append(image.id)

        for label in ann.labels:
            for tag in label.tags:
                if tag.name in self._tag_names:
                    tag_val = tag.value
                    curr_tag_data = self._name_to_data[tag.name]
                    idx = curr_tag_data[0][tag_val]
                    curr_tag_data[2][idx] += 1
                    if image.id not in curr_tag_data[1][idx]:
                        curr_tag_data[1][idx].append(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json(self) -> Optional[Dict]:
        if self._num_tags < 1:
            return None
        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
        }

        res = []

        for tag_name, curr_tag_stat_data in self._name_to_data.items():
            tag_val_names = list(curr_tag_stat_data[0].keys())
            colomns_options = [None] * (len(tag_val_names) + 1)
            colomns_options[0] = {"type": "class"}  # not used in Web

            curr_co_occurrence_matrix = curr_tag_stat_data[2]
            all_zeros = not curr_co_occurrence_matrix.any()
            if all_zeros:
                continue
            curr_references = curr_tag_stat_data[1]

            for idx in range(1, len(colomns_options)):
                colomns_options[idx] = {"maxValue": int(np.max(curr_co_occurrence_matrix[idx - 1]))}

            data = [tag_name] + curr_co_occurrence_matrix.tolist()

            curr_res = {
                "columns": ["Tag"] + tag_val_names,
                "rows": [tag_name],
                "data": [data],
                "referencesCell": {"0": curr_references},  # row - col
                "options": options,
                "colomnsOptions": colomns_options,
            }

            res.append(curr_res)

        if len(res) == 0:
            return None

        return res

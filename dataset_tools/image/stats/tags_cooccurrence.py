import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import HeatmapChart
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional
from itertools import groupby
import supervisely as sly
import numpy as np


from supervisely.imaging.color import random_rgb
import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

REFERENCES_LIMIT = 1000
IMAGES_TAGS = ["imagesOnly", "all"]
OBJECTS_TAGS = ["objectsOnly", "all"]
ONEOF_STRING = "oneof_string"
MAX_NUMBER_OF_COLUMNS = 100


def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class TagsCooccurrence(BaseStats):

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force

        self._name_to_index = {}
        self._sly_id_to_name = {}

        self._tag_name_to_type = {}
        self._tag_name_to_applicable = {}

        self._images_tags = []
        self._tag_names = []

        for tag_meta in self._meta.tag_metas:
            self._images_tags.append(tag_meta)

        for idx, im_tag_meta in enumerate(self._images_tags):
            self._name_to_index[im_tag_meta.name] = idx
            self._sly_id_to_name[im_tag_meta.sly_id] = im_tag_meta.name
            self._tag_name_to_type[im_tag_meta.name] = im_tag_meta.value_type
            self._tag_name_to_applicable[im_tag_meta.name] = im_tag_meta.applicable_to
            self._tag_names.append(im_tag_meta.name)

        self._references = defaultdict(lambda: defaultdict(list))

        self._num_tags = len(self._tag_names)
        self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

        self._tag_ids = {item.sly_id: item.name for item in self._meta.tag_metas}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]) -> None:
        if len(figures) == 0:
            return

        tags = set()
        for tag in image.tags:
            tag_name = self._tag_ids[tag["tagId"]]
            tags.add(tag_name)
        for figure in figures:
            for tag in figure.tags:
                tag_name = self._tag_ids[tag["tagId"]]
                tags.add(tag_name)

        for tag_name in tags:
            idx = self._name_to_index[tag_name]
            self.co_occurrence_matrix[idx][idx] += 1
            self._references[idx][idx].append(image.id)

        tags = list(tags)
        n = len(tags)
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = self._name_to_index[tags[i]]
                idx_j = self._name_to_index[tags[j]]
                self.co_occurrence_matrix[idx_i][idx_j] += 1
                self.co_occurrence_matrix[idx_j][idx_i] += 1

                self._references[idx_i][idx_j].append(image.id)
                self._references[idx_j][idx_i].append(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Optional[Dict]:
        if self._num_tags == 0:
            return None

        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview {tagType} tag applicable to {applicableTo}. {currentCell} images have objects of both tags {firstCell} and {currentColumn} at the same time",
        }
        colomns_options = [None] * (len(self._tag_names) + 1)
        colomns_options[0] = {"type": "tag"}  # not used in Web

        for idx in range(len(colomns_options) - 1):
            colomns_options[idx + 1] = {
                "tagType": self._tag_name_to_type[self._tag_names[idx]],
                "applicableTo": self._tag_name_to_applicable[self._tag_names[idx]],
            }

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
        if self._num_tags == 0:
            return

        matrix = np.array(self.co_occurrence_matrix, dtype="int32")
        n = self._num_tags
        ref_list = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ref_list[i][j] = set(self._references[i][j])

        references = np.array(ref_list, dtype=object)

        return np.stack([matrix, references], axis=0)

    def sew_chunks(self, chunks_dir: str, updated_classes: List[str] = []) -> np.ndarray:
        if self._num_tags == 0:
            return
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        res = np.zeros((self._num_tags, self._num_tags), dtype="int32")
        is_zero_area = None
        references = []

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
            array: np.ndarray, updated_classes: dict, insert_val=0
        ) -> Tuple[np.ndarray, np.ndarray]:
            if len(updated_classes) > 0:
                _updated_classes = list(updated_classes.values())
                indices = list(sorted([self._tag_names.index(cls) for cls in _updated_classes]))
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
                stat_data, ref_data = update_shape(loaded_data, updated_classes)

            res = np.add(stat_data, res)

            if len(references) == 0:
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


# class CooccurrenceImageTags(BaseStats):
#     """
#     Columns:
#         Class
#         class 1
#         class 2
#         etc.
#     """

#     def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
#         self._meta = project_meta
#         self._stats = {}
#         self.force = force

#         self._name_to_index = {}
#         self._sly_id_to_name = {}

#         self._tag_name_to_type = {}

#         self._images_tags = []
#         self._tag_names = []

#         for tag_meta in self._meta.tag_metas:
#             if tag_meta.applicable_to in IMAGES_TAGS:
#                 self._images_tags.append(tag_meta)

#         for idx, im_tag_meta in enumerate(self._images_tags):
#             self._name_to_index[im_tag_meta.name] = idx
#             self._sly_id_to_name[im_tag_meta.sly_id] = im_tag_meta.name
#             self._tag_name_to_type[im_tag_meta.name] = im_tag_meta.value_type
#             self._tag_names.append(im_tag_meta.name)

#         self._references = defaultdict(lambda: defaultdict(list))

#         self._num_tags = len(self._tag_names)
#         self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

#     def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

#         image_tags = set()
#         for tag_info in image.tags:
#             image_tags.add(self._sly_id_to_name[tag_info["tagId"]])

#         for image_tag_ in image_tags:
#             idx = self._name_to_index[image_tag_]
#             self.co_occurrence_matrix[idx][idx] += 1
#             self._references[idx][idx].append(image.id)

#         image_tags = list(image_tags)
#         for i in range(len(image_tags)):
#             for j in range(i + 1, len(image_tags)):
#                 tag_i = image_tags[i]
#                 tag_j = image_tags[j]
#                 idx_i = self._name_to_index[tag_i]
#                 idx_j = self._name_to_index[tag_j]
#                 self.co_occurrence_matrix[idx_i][idx_j] += 1
#                 self.co_occurrence_matrix[idx_j][idx_i] += 1

#                 if len(self._references[idx_i][idx_j]) <= REFERENCES_LIMIT:
#                     self._references[idx_i][idx_j].append(image.id)
#                 if len(self._references[idx_j][idx_i]) <= REFERENCES_LIMIT:
#                     self._references[idx_j][idx_i].append(image.id)

#     def clean(self) -> None:
#         self.__init__(self._meta, self.force)

#     def to_json(self) -> Optional[Dict]:
#         if self._num_tags <= 1:
#             return None
#         options = {
#             "fixColumns": 1,  # not used in Web
#             "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
#         }
#         colomns_options = [None] * (len(self._tag_names) + 1)
#         colomns_options[0] = {"type": "class"}  # not used in Web

#         for idx in range(len(colomns_options) - 1):
#             colomns_options[idx + 1] = {
#                 "subtitle": self._tag_name_to_type[self._tag_names[idx]],
#             }
#             # colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

#         data = [
#             [value] + sublist
#             for value, sublist in zip(self._tag_names, self.co_occurrence_matrix.tolist())
#         ]

#         res = {
#             "columns": ["Tag"] + self._tag_names,
#             "data": data,
#             "referencesCell": self._references,  # row - col
#             "options": options,
#             "colomnsOptions": colomns_options,
#         }
#         return res


# class CooccurrenceObjectTags(BaseStats):

#     def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
#         self._meta = project_meta
#         self._stats = {}
#         self.force = force

#         self._name_to_index = {}
#         self._tag_names = []
#         self._tag_name_to_type = {}

#         self._object_tags = []

#         for tag_meta in self._meta.tag_metas:
#             if tag_meta.applicable_to in OBJECTS_TAGS:
#                 self._object_tags.append(tag_meta)

#         for idx, obj_tag_meta in enumerate(self._object_tags):
#             self._name_to_index[obj_tag_meta.name] = idx
#             self._tag_name_to_type[obj_tag_meta.name] = obj_tag_meta.value_type
#             self._tag_names.append(obj_tag_meta.name)

#         self._references = defaultdict(lambda: defaultdict(list))

#         self._num_tags = len(self._tag_names)
#         self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

#     def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

#         for label in ann.labels:
#             label_tags = set()

#             for tag in label.tags:
#                 if tag.name in self._tag_names:
#                     label_tag_ = tag.name
#                     label_tags.add(tag.name)

#             for label_tag_ in label_tags:
#                 idx = self._name_to_index[label_tag_]
#                 self.co_occurrence_matrix[idx][idx] += 1
#                 if image.id not in self._references[idx][idx]:
#                     self._references[idx][idx].append(image.id)

#             label_tags = list(label_tags)
#             for i in range(len(label_tags)):
#                 for j in range(i + 1, len(label_tags)):
#                     tag_i = label_tags[i]
#                     tag_j = label_tags[j]
#                     idx_i = self._name_to_index[tag_i]
#                     idx_j = self._name_to_index[tag_j]
#                     self.co_occurrence_matrix[idx_i][idx_j] += 1
#                     self.co_occurrence_matrix[idx_j][idx_i] += 1

#                     if len(self._references[idx_i][idx_j]) <= REFERENCES_LIMIT:
#                         if image.id not in self._references[idx_i][idx_j]:
#                             self._references[idx_i][idx_j].append(image.id)
#                     if len(self._references[idx_j][idx_i]) <= REFERENCES_LIMIT:
#                         if image.id not in self._references[idx_j][idx_i]:
#                             self._references[idx_j][idx_i].append(image.id)

#     def clean(self) -> None:
#         self.__init__(self._meta, self.force)

#     def to_json(self) -> Optional[Dict]:
#         if self._num_tags < 1:
#             return None
#         options = {
#             "fixColumns": 1,  # not used in Web
#             "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
#         }
#         colomns_options = [None] * (len(self._tag_names) + 1)
#         colomns_options[0] = {"type": "class"}  # not used in Web

#         for idx in range(len(colomns_options) - 1):
#             colomns_options[idx + 1] = {
#                 "subtitle": self._tag_name_to_type[self._tag_names[idx]],
#             }
#             # colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

#         data = [
#             [value] + sublist
#             for value, sublist in zip(self._tag_names, self.co_occurrence_matrix.tolist())
#         ]

#         res = {
#             "columns": ["Tag"] + self._tag_names,
#             "data": data,
#             "referencesCell": self._references,  # row - col
#             "options": options,
#             "colomnsOptions": colomns_options,
#         }
#         return res


class OneOfTagsDistribution(BaseStats):

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force

        # self._name_to_index = {}
        # self._name_to_data = {}
        # self._sly_id_to_name = {}

        # self._tag_name_to_type = {}
        # self._tag_names = []

        # for idx, tag_meta in enumerate(self._meta.tag_metas):
        #     if tag_meta.value_type == ONEOF_STRING:
        #         self._tag_names.append(tag_meta.name)
        #         curr_val_to_index = {}
        #         for idx, possible_val in enumerate(tag_meta.possible_values):
        #             curr_val_to_index[possible_val] = idx
        #             curr_references = defaultdict(list)
        #             curr_co_occurrence_matrix = np.zeros((len(tag_meta.possible_values)), dtype=int)

        #         self._name_to_data[tag_meta.name] = (
        #             curr_val_to_index,
        #             curr_references,
        #             curr_co_occurrence_matrix,
        #         )

        #         self._sly_id_to_name[tag_meta.sly_id] = tag_meta.name
        # self._num_tags = len(list(self._name_to_data.keys()))

        # new
        self._tag_ids = defaultdict(str)
        self._id_to_data = {}

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            if tag_meta.value_type == ONEOF_STRING:
                self._tag_ids[tag_meta.sly_id] = tag_meta.name
                curr_val_to_index = {}
                for idx, possible_val in enumerate(tag_meta.possible_values):
                    curr_val_to_index[possible_val] = idx
                    curr_references = defaultdict(list)
                    curr_co_occurrence_matrix = np.zeros((len(tag_meta.possible_values)), dtype=int)

                self._id_to_data[tag_meta.sly_id] = (
                    curr_val_to_index,
                    curr_references,
                    curr_co_occurrence_matrix,
                )
                # self._sly_id_to_name[tag_meta.sly_id] = tag_meta.name
        # self._tag_ids = {
        #     item.sly_id: item.name for item in self._meta.tag_metas if item.name in self._tag_names
        # }
        self._num_tags = len(list(self._tag_ids))

        # self._distribution_dict = {class_id: {0: set()} for class_id in self._tag_ids}
        self._max_count = 0
        self._classes_hex = {item.sly_id: rgb_to_hex(item.color) for item in self._meta.obj_classes}

    # def update2(self, image: ImageInfo, figures: List[FigureInfo]):

    #     for tag in image.tags:
    #         if tag["tagId"] in self._tag_ids:
    #             tag_val = tag["value"]
    #             curr_tag_data = self._id_to_data[tag["tagId"]]
    #             idx = curr_tag_data[0][tag_val]
    #             curr_tag_data[2][idx] += 1
    #             if image.id not in curr_tag_data[1][idx]:
    #                 curr_tag_data[1][idx].append(image.id)

    #     for figure in figures:
    #         for tag in figure.tags:
    #             if tag.name in self._tag_names:
    #                 tag_val = tag.value
    #                 curr_tag_data = self._name_to_data[tag.name]
    #                 idx = curr_tag_data[0][tag_val]
    #                 curr_tag_data[2][idx] += 1
    #                 if image.id not in curr_tag_data[1][idx]:
    #                     curr_tag_data[1][idx].append(image.id)

    # def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

    #     for tag_info in image.tags:
    #         tag_name = self._sly_id_to_name.get(tag_info["tagId"])
    #         if tag_name in self._tag_names:
    #             tag_val = tag_info["value"]
    #             curr_tag_data = self._name_to_data[tag_name]
    #             idx = curr_tag_data[0][tag_val]
    #             curr_tag_data[2][idx] += 1
    #             if image.id not in curr_tag_data[1][idx]:
    #                 curr_tag_data[1][idx].append(image.id)

    #     for label in ann.labels:
    #         for tag in label.tags:
    #             if tag.name in self._tag_names:
    #                 tag_val = tag.value
    #                 curr_tag_data = self._name_to_data[tag.name]
    #                 idx = curr_tag_data[0][tag_val]
    #                 curr_tag_data[2][idx] += 1
    #                 if image.id not in curr_tag_data[1][idx]:
    #                     curr_tag_data[1][idx].append(image.id)

    # def clean(self) -> None:
    #     self.__init__(self._meta, self.force)

    # def to_json2(self) -> Dict:
    #     series, colors = [], []
    #     references = defaultdict(dict)
    #     axis = [i for i in range(self._max_count + 1)]
    #     for class_id, class_name in self._class_ids.items():
    #         class_ditrib = self._distribution_dict[class_id]
    #         reference = {x: [] for x in axis}

    #         values = [0 for _ in range(self._max_count + 1)]
    #         for objects_count, images_set in class_ditrib.items():
    #             values[objects_count] = len(images_set)

    #             seized_refs = self._seize_list_to_fixed_size(list(images_set), REFERENCES_LIMIT)
    #             reference[objects_count] = seized_refs
    #             references.setdefault(class_name, {}).update(reference)

    #         row = {
    #             "name": class_name,
    #             "y": values,
    #             "x": axis,
    #         }

    #         series.append(row)
    #         colors.append(self._classes_hex[class_id])

    #     hmp = HeatmapChart(
    #         title="Objects on images - distribution for every class",
    #         xaxis_title="Number of objects on image",
    #         color_range="row",
    #         tooltip="Click to preview {y} images with {x} objects of class {series_name}",
    #     )
    #     hmp.add_series_batch(series)

    #     number_of_rows = len(series)
    #     max_widget_height = 10000
    #     if number_of_rows < 5:
    #         row_height = 70
    #     elif number_of_rows < 20:
    #         row_height = 50
    #     else:
    #         row_height = 30

    #     res = hmp.get_json_data()
    #     number_of_columns = len(axis)
    #     calculated_height = number_of_rows * row_height
    #     height = min(calculated_height, max_widget_height) + 150
    #     res["referencesCell"] = references
    #     res["options"]["chart"]["height"] = height
    #     res["options"]["colors"] = colors

    #     # Disabling labels and ticks for x-axis if there are too many columns.
    #     if MAX_NUMBER_OF_COLUMNS > number_of_columns > 40:
    #         res["options"]["xaxis"]["labels"] = {"show": False}
    #         res["options"]["xaxis"]["axisTicks"] = {"show": False}
    #         res["options"]["dataLabels"] = {"enabled": False}
    #     elif number_of_columns >= MAX_NUMBER_OF_COLUMNS:
    #         return

    #     return res

    # def to_json(self) -> Optional[Dict]:
    #     if self._num_tags < 1:
    #         return None
    #     options = {
    #         "fixColumns": 1,  # not used in Web
    #         "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at the same time",
    #     }

    #     res = []

    #     for tag_name, curr_tag_stat_data in self._name_to_data.items():
    #         tag_val_names = list(curr_tag_stat_data[0].keys())
    #         colomns_options = [None] * (len(tag_val_names) + 1)
    #         colomns_options[0] = {"type": "class"}  # not used in Web

    #         curr_co_occurrence_matrix = curr_tag_stat_data[2]
    #         all_zeros = not curr_co_occurrence_matrix.any()
    #         if all_zeros:
    #             continue
    #         curr_references = curr_tag_stat_data[1]

    #         for idx in range(1, len(colomns_options)):
    #             colomns_options[idx] = {"maxValue": int(np.max(curr_co_occurrence_matrix[idx - 1]))}

    #         data = [tag_name] + curr_co_occurrence_matrix.tolist()

    #         curr_res = {
    #             "columns": ["Tag"] + tag_val_names,
    #             "rows": [tag_name],
    #             "data": [data],
    #             "referencesCell": {"0": curr_references},  # row - col
    #             "options": options,
    #             "colomnsOptions": colomns_options,
    #         }

    #         res.append(curr_res)

    #     if len(res) == 0:
    #         return None

    #     return res

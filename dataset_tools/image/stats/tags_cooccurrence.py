import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import HeatmapChart, Apexchart
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


class TagsImagesCooccurrence(BaseStats):

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
            if tag_meta.applicable_to in IMAGES_TAGS:
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
            "cellTooltip": "Click to preview. {currentCell} images have both tags {firstCell} and {currentColumn} at same time",
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


class TagsObjectsCooccurrence(BaseStats):

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
            if tag_meta.applicable_to in OBJECTS_TAGS:
                self._images_tags.append(tag_meta)

        for idx, im_tag_meta in enumerate(self._images_tags):
            self._name_to_index[im_tag_meta.name] = idx
            self._sly_id_to_name[im_tag_meta.sly_id] = im_tag_meta.name
            self._tag_name_to_type[im_tag_meta.name] = im_tag_meta.value_type
            self._tag_name_to_applicable[im_tag_meta.name] = im_tag_meta.applicable_to
            self._tag_names.append(im_tag_meta.name)

        self._tag_name_to_index = {}
        for idx, tag_name in enumerate(self._tag_names):
            self._tag_name_to_index[tag_name] = idx

        self._references = defaultdict(lambda: defaultdict(set))

        self._num_tags = len(self._tag_names)
        self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

        self._tag_ids = {item.sly_id: item.name for item in self._meta.tag_metas}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]) -> None:
        if len(figures) == 0:
            return

        for figure in figures:
            tag_names = [self._tag_ids[tag["tagId"]] for tag in figure.tags]
            for tag_name in tag_names:
                idx = self._name_to_index[tag_name]
                self.co_occurrence_matrix[idx][idx] += 1
                self._references[idx][idx].add(image.id)

            n = len(tag_names)
            for i in range(n):
                for j in range(i + 1, n):
                    idx_i = self._name_to_index[tag_names[i]]
                    idx_j = self._name_to_index[tag_names[j]]

                    self.co_occurrence_matrix[idx_i][idx_j] += 1
                    self.co_occurrence_matrix[idx_j][idx_i] += 1

                    self._references[idx_i][idx_j].add(image.id)
                    self._references[idx_j][idx_i].add(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Optional[Dict]:
        if self._num_tags == 0:
            return None

        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. {currentCell} objects have both tags {firstCell} and {currentColumn} at same time",
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
        # references = defaultdict(lambda: defaultdict(list))
        references_count = np.zeros((self._num_tags, self._num_tags), dtype=int)
        for rows in self._references:
            for col in self._references[rows]:
                references_count[rows][col] = len(self._references[rows][col])

        references_count_data = [
            [value] + sublist for value, sublist in zip(self._tag_names, references_count.tolist())
        ]

        res = {
            "columns": ["Tag"] + self._tag_names,
            "data": data,
            "referencesCellCount": references_count_data,
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


class TagsImagesOneOfDistribution(BaseStats):

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force

        # new
        self._tag_ids = defaultdict(str)
        self._tag_vals = defaultdict(list)
        self._id_to_data = {}

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            if tag_meta.value_type == ONEOF_STRING and tag_meta.applicable_to in IMAGES_TAGS:
                self._tag_ids[tag_meta.sly_id] = tag_meta.name
                self._tag_vals[tag_meta.sly_id] = tag_meta.possible_values

        self._num_tags = len(list(self._tag_ids))
        self._tag_ids = dict(sorted(self._tag_ids.items(), key=lambda item: item[1]))

        self._max_count = 0
        for vals in self._tag_vals.values():
            self._max_count = max(self._max_count, len(vals))

        self._objects_cnt_dict = defaultdict(lambda: defaultdict(int))
        self._references_dict = defaultdict(lambda: defaultdict(set))

        for tag_id in self._tag_ids:
            for val in self._tag_vals[tag_id]:
                self._objects_cnt_dict[tag_id][val] = 0
                self._references_dict[tag_id][val] = set()

        self._tags_hex = {item.sly_id: rgb_to_hex(item.color) for item in self._meta.tag_metas}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        _tags_oneof = []
        for tag in image.tags:
            if tag["tagId"] in self._tag_ids:
                _tags_oneof.append((tag["tagId"], tag["value"]))

        for tag_id, val in _tags_oneof:
            self._objects_cnt_dict[tag_id][val] += 1
            self._references_dict[tag_id][val].add(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Dict:
        if len(self._tag_ids) == 0:
            return
        series, colors = [], []
        references = defaultdict(dict)
        axis = [i for i in range(self._max_count)]
        for tag_id, tag_name in self._tag_ids.items():
            values = [x for x in self._objects_cnt_dict[tag_id].values()]
            reference = {x: [] for x in axis}
            if len(values) < len(axis):
                values += [-1] * (len(axis) - len(values))

            for idx, images in enumerate(self._references_dict[tag_id].values()):
                reference[idx] = list(images)
                references.setdefault(tag_name, {}).update(reference)

            row = {
                "name": tag_name,
                "y": values,
                "x": axis,
            }

            series.append(row)
            colors.append(self._tags_hex[tag_id])

        hmp = HeatmapChart(
            title="Images",
            color_range="row",
            tooltip="Click to preview {y} images with tag {series_name} and value {tag_value}",
        )
        hmp.add_series_batch(series)

        number_of_rows = len(series)
        max_widget_height = 10000
        if number_of_rows < 5:
            row_height = 70
        elif number_of_rows < 20:
            row_height = 50
        else:
            row_height = 30

        res = hmp.get_json_data()
        _tags = dict(self._objects_cnt_dict).values()
        for series, _t in zip(res["series"], _tags):
            expand_t = list(dict(_t))
            if len(_t) < len(series["data"]):
                delta = len(series["data"]) - len(_t)
                expand_t += [""] * delta

            for data, title in zip(series["data"], expand_t):
                data["title"] = title

        for item in res["series"]:
            item["data"] = sorted(item["data"], key=lambda x: x["y"], reverse=True)

        number_of_columns = len(axis)
        calculated_height = number_of_rows * row_height
        height = min(calculated_height, max_widget_height) + 150
        res["referencesCell"] = references
        res["options"]["chart"]["height"] = height
        res["options"]["colors"] = colors

        res["options"]["xaxis"]["axisTicks"] = {"show": False}
        res["options"]["xaxis"]["hideXaxis"] = {"hide": True}

        res["options"]["plotOptions"]["heatmap"]["colorScale"]["ranges"][0]["color"] = "#A5A5A5"
        res["options"]["plotOptions"]["heatmap"]["colorScale"]["ranges"].append(
            {"from": -1, "to": -1, "name": "", "color": "#FFFFFF"}
        )

        # Disabling labels and ticks for x-axis if there are too many columns.
        if MAX_NUMBER_OF_COLUMNS > number_of_columns > 40:
            res["options"]["xaxis"]["labels"] = {"show": False}
            res["options"]["xaxis"]["axisTicks"] = {"show": False}
            res["options"]["dataLabels"] = {"enabled": False}
        elif number_of_columns >= MAX_NUMBER_OF_COLUMNS:
            return

        return res

    def to_numpy_raw(self):
        _data = dict(self._objects_cnt_dict)
        _refs = dict(self._references_dict)
        for tag_id, vals in dict(_data).items():
            _data[tag_id] = dict(vals)
            for val, obj_cnt in dict(vals).items():
                _data[tag_id][val] = [obj_cnt, _refs[tag_id][val]]

        return np.array(_data, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str, updated_classes: dict) -> np.ndarray:

        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data:
                loaded_tags = set([tag_id for tag_id in loaded_data])
                true_tags = set(self._tag_ids)

                added = true_tags - loaded_tags
                for tag_id in list(added):
                    vals = self._tag_vals[tag_id]
                    if loaded_data.get(tag_id) is None:
                        loaded_data[tag_id] = {}
                        for val in vals:
                            loaded_data[tag_id][val] = [0, set()]
                    for val in vals:
                        if loaded_data[tag_id].get(val) is None:
                            loaded_data[tag_id][val] = [0, set()]

                removed = loaded_tags - true_tags
                for tag_id in list(removed):
                    loaded_data.pop(tag_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for tag_id in self._tag_ids:
                    vals = self._tag_vals[tag_id]

                    if loaded_data.get(tag_id) is None:
                        loaded_data[tag_id] = {}
                        for val in vals:
                            loaded_data[tag_id][val] = [0, set()]
                    for val in vals:
                        if loaded_data[tag_id].get(val) is None:
                            loaded_data[tag_id][val] = [0, set()]

                    for val in vals:
                        obj_cnt = loaded_data[tag_id][val][0]
                        self._objects_cnt_dict[tag_id][val] += obj_cnt
                        images_set = loaded_data[tag_id][val][1]
                        self._references_dict[tag_id][val].update(images_set)

        return None


class TagsObjectsOneOfDistribution(BaseStats):

    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
        self._meta = project_meta
        self._stats = {}
        self.force = force

        # new
        self._tag_ids = defaultdict(str)
        self._tag_vals = defaultdict(list)
        self._id_to_data = {}

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            if tag_meta.value_type == ONEOF_STRING and tag_meta.applicable_to in OBJECTS_TAGS:
                self._tag_ids[tag_meta.sly_id] = tag_meta.name
                self._tag_vals[tag_meta.sly_id] = tag_meta.possible_values

        self._num_tags = len(list(self._tag_ids))
        self._tag_ids = dict(sorted(self._tag_ids.items(), key=lambda item: item[1]))

        self._objects_cnt_dict = defaultdict(lambda: defaultdict(int))
        self._references_dict = defaultdict(lambda: defaultdict(set))

        for tag_id in self._tag_ids:
            for val in self._tag_vals[tag_id]:
                self._objects_cnt_dict[tag_id][val] = 0
                self._references_dict[tag_id][val] = set()

        self._max_count = 0
        for vals in self._tag_vals.values():
            self._max_count = max(self._max_count, len(vals))
        self._tags_hex = {item.sly_id: rgb_to_hex(item.color) for item in self._meta.tag_metas}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        _tags_oneof = []

        for figure in figures:
            for tag in figure.tags:
                if tag["tagId"] in self._tag_ids:
                    _tags_oneof.append((tag["tagId"], tag["value"]))

        for tag_id, val in _tags_oneof:
            self._objects_cnt_dict[tag_id][val] += 1
            self._references_dict[tag_id][val].add(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Dict:
        if len(self._tag_ids) == 0:
            return
        series, colors = [], []
        references = defaultdict(dict)
        axis = [i for i in range(self._max_count)]
        for tag_id, tag_name in self._tag_ids.items():
            values = [x for x in self._objects_cnt_dict[tag_id].values()]
            reference = {x: [] for x in axis}
            if len(values) < len(axis):
                values += [-1] * (len(axis) - len(values))

            for idx, images in enumerate(self._references_dict[tag_id].values()):
                reference[idx] = list(images)
                tag_name = self._tag_ids[tag_id]
                references.setdefault(tag_name, {}).update(reference)

            row = {
                "name": tag_name,
                "y": values,
                "x": axis,
            }

            series.append(row)
            colors.append(self._tags_hex[tag_id])

        hmp = HeatmapChart(
            title="Objects",
            color_range="row",
            tooltip="Click to preview {y} images with tag {series_name} and value {tag_value}",
        )
        hmp.add_series_batch(series)

        number_of_rows = len(series)
        max_widget_height = 10000
        if number_of_rows < 5:
            row_height = 70
        elif number_of_rows < 20:
            row_height = 50
        else:
            row_height = 30

        res = hmp.get_json_data()
        _tags = dict(self._objects_cnt_dict).values()
        for series, _t in zip(res["series"], _tags):
            expand_t = list(dict(_t))
            if len(_t) < len(series["data"]):
                delta = len(series["data"]) - len(_t)
                expand_t += [""] * delta

            for data, title in zip(series["data"], expand_t):
                data["title"] = title

        for item in res["series"]:
            item["data"] = sorted(item["data"], key=lambda x: x["y"], reverse=True)

        number_of_columns = len(axis)
        calculated_height = number_of_rows * row_height
        height = min(calculated_height, max_widget_height) + 150
        res["referencesCell"] = references
        res["options"]["chart"]["height"] = height
        res["options"]["colors"] = colors

        res["options"]["xaxis"]["axisTicks"] = {"show": False}
        res["options"]["xaxis"]["hideXaxis"] = {"hide": True}

        res["options"]["plotOptions"]["heatmap"]["colorScale"]["ranges"][0]["color"] = "#A5A5A5"
        res["options"]["plotOptions"]["heatmap"]["colorScale"]["ranges"].append(
            {"from": -1, "to": -1, "name": "", "color": "#FFFFFF"}
        )

        # Disabling labels and ticks for x-axis if there are too many columns.
        if MAX_NUMBER_OF_COLUMNS > number_of_columns > 40:
            res["options"]["xaxis"]["labels"] = {"show": False}
            res["options"]["xaxis"]["axisTicks"] = {"show": False}
            res["options"]["dataLabels"] = {"enabled": False}
        elif number_of_columns >= MAX_NUMBER_OF_COLUMNS:
            return

        return res

    def to_numpy_raw(self):
        _data = dict(self._objects_cnt_dict)
        _refs = dict(self._references_dict)
        for tag_id, vals in dict(_data).items():
            _data[tag_id] = dict(vals)
            for val, obj_cnt in dict(vals).items():
                _data[tag_id][val] = [obj_cnt, _refs[tag_id][val]]

        return np.array(_data, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str, updated_classes: dict) -> np.ndarray:

        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data:
                loaded_tags = set([tag_id for tag_id in loaded_data])
                true_tags = set(self._tag_ids)

                added = true_tags - loaded_tags
                for tag_id in list(added):
                    vals = self._tag_vals[tag_id]
                    if loaded_data.get(tag_id) is None:
                        loaded_data[tag_id] = {}
                        for val in vals:
                            loaded_data[tag_id][val] = [0, set()]
                    for val in vals:
                        if loaded_data[tag_id].get(val) is None:
                            loaded_data[tag_id][val] = [0, set()]

                removed = loaded_tags - true_tags
                for tag_id in list(removed):
                    loaded_data.pop(tag_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for tag_id in self._tag_ids:
                    vals = self._tag_vals[tag_id]

                    if loaded_data.get(tag_id) is None:
                        loaded_data[tag_id] = {}
                        for val in vals:
                            loaded_data[tag_id][val] = [0, set()]
                    for val in vals:
                        if loaded_data[tag_id].get(val) is None:
                            loaded_data[tag_id][val] = [0, set()]

                    for val in vals:
                        obj_cnt = loaded_data[tag_id][val][0]
                        self._objects_cnt_dict[tag_id][val] += obj_cnt
                        images_set = loaded_data[tag_id][val][1]
                        self._references_dict[tag_id][val].update(images_set)

        return None

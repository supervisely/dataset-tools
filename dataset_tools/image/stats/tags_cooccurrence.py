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
MAX_NUMBER_OF_TAGS = 500


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
        self._tag_ids = {}

        self._num_tags = len(self._meta.tag_metas)
        self.co_occurrence_dict = {}
        if self._num_tags > MAX_NUMBER_OF_TAGS:
            return

        for tag_meta in self._meta.tag_metas:
            if tag_meta.applicable_to in IMAGES_TAGS:
                self._images_tags.append(tag_meta)
                self._tag_ids[tag_meta.sly_id] = tag_meta.name

        for idx, im_tag_meta in enumerate(self._images_tags):
            self._name_to_index[im_tag_meta.name] = idx
            self._sly_id_to_name[im_tag_meta.sly_id] = im_tag_meta.name
            self._tag_name_to_type[im_tag_meta.name] = im_tag_meta.value_type
            self._tag_name_to_applicable[im_tag_meta.name] = im_tag_meta.applicable_to
            self._tag_names.append(im_tag_meta.name)

        # self._references = defaultdict(lambda: defaultdict(list))

        # self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

        self.co_occurrence_dict = {
            class_id_x: {class_id_y: set() for class_id_y in self._tag_ids}
            for class_id_x in self._tag_ids
        }

        # self._tag_ids = {item.sly_id: item.name for item in self._meta.tag_metas}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]) -> None:
        if len(image.tags) == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return

        tags = set()
        for tag in image.tags:
            tags.add(tag["tagId"])

        for tag_id in tags:
            if tag_id not in self.co_occurrence_dict:
                self.co_occurrence_dict[tag_id] = {tag_id: set()}
            self.co_occurrence_dict[tag_id][tag_id].add(image.id)

        for tag_id_i in tags:
            for tag_id_j in tags:
                if tag_id_j not in self.co_occurrence_dict[tag_id_i]:
                    self.co_occurrence_dict[tag_id_i][tag_id_j] = set()
                self.co_occurrence_dict[tag_id_i][tag_id_j].add(image.id)
                
                if tag_id_i not in self.co_occurrence_dict[tag_id_j]:
                    self.co_occurrence_dict[tag_id_j][tag_id_i] = set()
                self.co_occurrence_dict[tag_id_j][tag_id_i].add(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Optional[Dict]:
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
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

        keys = list(self._tag_ids)
        index = {key: idx for idx, key in enumerate(keys)}
        size = self._num_tags
        nested_list = [[None] * size for _ in range(size)]
        refs_image_ids = {x: {y: None for y in range(size)} for x in range(size)}

        for row_key, subdict in self.co_occurrence_dict.items():
            for col_key, image_ids in subdict.items():
                row_idx = index[row_key]
                col_idx = index[col_key]
                nested_list[row_idx][col_idx] = len(image_ids)
                nested_list[col_idx][row_idx] = len(image_ids)
                refs_image_ids[row_idx][col_idx] = list(image_ids)
                refs_image_ids[col_idx][row_idx] = list(image_ids)

        data = [[value] + sublist for value, sublist in zip(self._tag_names, nested_list)]

        res = {
            "columns": ["Tag"] + self._tag_names,
            "data": data,
            "referencesCell": refs_image_ids,
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return
        return np.array(self.co_occurrence_dict, dtype=object)

    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return

        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                loaded_tags = set(loaded_data)
                true_tags = set(self._tag_ids)

                added = true_tags - loaded_tags
                for tag_id_new in added:
                    loaded_data[tag_id_new] = dict()
                    loaded_data[tag_id_new][tag_id_new] = set()
                    for tag_id_old in loaded_tags:
                        loaded_data[tag_id_new][tag_id_old] = set()
                        loaded_data[tag_id_old][tag_id_new] = set()

                removed = loaded_tags - true_tags
                for tag_id_rm in removed:
                    loaded_data.pop(tag_id_rm)
                    for tag_id in true_tags:
                        loaded_data[tag_id].pop(tag_id_rm)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for tag_id_i in true_tags:
                    for tag_id_j in true_tags:
                        self.co_occurrence_dict[tag_id_i][tag_id_j].update(
                            loaded_data[tag_id_i].get(tag_id_j, set())
                        )
                        self.co_occurrence_dict[tag_id_j][tag_id_i].update(
                            loaded_data[tag_id_j].get(tag_id_i, set())
                        )


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

        self._tag_ids = {}  # item.sly_id: item.name for item in self._meta.tag_metas}

        self._num_tags = len(self._meta.tag_metas)
        self.co_occurrence_dict = {}
        if self._num_tags > MAX_NUMBER_OF_TAGS:
            return

        for tag_meta in self._meta.tag_metas:
            if tag_meta.applicable_to in OBJECTS_TAGS:
                self._images_tags.append(tag_meta)
                self._tag_ids[tag_meta.sly_id] = tag_meta.name

        for idx, im_tag_meta in enumerate(self._images_tags):
            self._name_to_index[im_tag_meta.name] = idx
            self._sly_id_to_name[im_tag_meta.sly_id] = im_tag_meta.name
            self._tag_name_to_type[im_tag_meta.name] = im_tag_meta.value_type
            self._tag_name_to_applicable[im_tag_meta.name] = im_tag_meta.applicable_to
            self._tag_names.append(im_tag_meta.name)

        self._tag_name_to_index = {}
        for idx, tag_name in enumerate(self._tag_names):
            self._tag_name_to_index[tag_name] = idx

        # self._references = defaultdict(lambda: defaultdict(set))

        self._num_tags = len(self._tag_names)
        # self.co_occurrence_matrix = np.zeros((self._num_tags, self._num_tags), dtype=int)

        self.co_occurrence_dict = {
            tag_id_x: {tag_id_y: set() for tag_id_y in self._tag_ids} for tag_id_x in self._tag_ids
        }
        self.references_dict = {
            tag_id_x: {tag_id_y: set() for tag_id_y in self._tag_ids} for tag_id_x in self._tag_ids
        }
        # self._tag_ids = {item.sly_id: item.name for item in self._meta.tag_metas}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]) -> None:
        if len(figures) == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return

        for figure in figures:
            tags = set()
            for tag in figure.tags:
                tags.add(tag["tagId"])

            for tag_id in tags:
                self.co_occurrence_dict[tag_id][tag_id].add(figure.id)
                self.references_dict[tag_id][tag_id].add(image.id)

            for tag_id_i in tags:
                for tag_id_j in tags:
                    self.co_occurrence_dict[tag_id_i][tag_id_j].add(figure.id)
                    self.co_occurrence_dict[tag_id_j][tag_id_i].add(figure.id)
                    self.references_dict[tag_id_i][tag_id_j].add(image.id)
                    self.references_dict[tag_id_j][tag_id_i].add(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Optional[Dict]:
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return None

        options = {
            "fixColumns": 1,
            "cellTooltip": "Click to preview. {currentCell} objects have both tags {firstCell} and {currentColumn} at same time",
        }
        colomns_options = [None] * (len(self._tag_names) + 1)
        colomns_options[0] = {"type": "tag"}

        for idx in range(len(colomns_options) - 1):
            colomns_options[idx + 1] = {
                "tagType": self._tag_name_to_type[self._tag_names[idx]],
                "applicableTo": self._tag_name_to_applicable[self._tag_names[idx]],
            }

        keys = list(self._tag_ids)
        index = {key: idx for idx, key in enumerate(keys)}
        size = self._num_tags
        nested_objects = [[None] * size for _ in range(size)]
        nested_images = [[None] * size for _ in range(size)]
        refs_image_ids = {x: {y: None for y in range(size)} for x in range(size)}

        for row_key, subdict in self.co_occurrence_dict.items():
            for col_key, object_ids in subdict.items():
                row_idx = index[row_key]
                col_idx = index[col_key]
                nested_objects[row_idx][col_idx] = len(object_ids)
                nested_objects[col_idx][row_idx] = len(object_ids)

        for row_key, subdict in self.references_dict.items():
            for col_key, image_ids in subdict.items():
                row_idx = index[row_key]
                col_idx = index[col_key]
                nested_images[row_idx][col_idx] = len(image_ids)
                nested_images[col_idx][row_idx] = len(image_ids)
                refs_image_ids[row_idx][col_idx] = list(image_ids)
                refs_image_ids[col_idx][row_idx] = list(image_ids)

        data = [[value] + sublist for value, sublist in zip(self._tag_names, nested_objects)]
        data_refs = [[value] + sublist for value, sublist in zip(self._tag_names, nested_images)]

        res = {
            "columns": ["Tag"] + self._tag_names,
            "data": data,
            "referencesCellCount": data_refs,
            "referencesCell": refs_image_ids,
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return
        return np.array(
            {"objects": self.co_occurrence_dict, "images": self.references_dict}, dtype=object
        )

    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return

        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                loaded_tags = set(loaded_data["objects"].keys())
                true_tags = set(self._tag_ids.keys())
                added = true_tags - loaded_tags
                removed = loaded_tags - true_tags

                for data, res in zip(
                    (loaded_data["objects"], loaded_data["images"]),
                    (self.co_occurrence_dict, self.references_dict),
                ):

                    for tag_id_new in added:
                        data[tag_id_new] = dict()
                        data[tag_id_new][tag_id_new] = set()
                        for tag_id_old in loaded_tags:
                            data[tag_id_new][tag_id_old] = set()
                            data[tag_id_old][tag_id_new] = set()

                    for tag_id_rm in removed:
                        data.pop(tag_id_rm)
                        for tag_id in true_tags:
                            data[tag_id].pop(tag_id_rm)

                    for tag_id_i in true_tags:
                        for tag_id_j in true_tags:
                            res[tag_id_i][tag_id_j].update(data[tag_id_i].get(tag_id_j, set()))
                            res[tag_id_j][tag_id_i].update(data[tag_id_j].get(tag_id_i, set()))

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)


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
        if len(image.tags) == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return

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
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
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
        _tags_objcnt = dict(self._objects_cnt_dict).values()
        _tags_refscnt = dict(self._references_dict).values()
        for series, _objcnt, _refscnt in zip(res["series"], _tags_objcnt, _tags_refscnt):
            expand_obj = list(_objcnt)
            expand_ref = [len(x) for x in _refscnt.values()]
            if len(_objcnt) < len(series["data"]):
                delta = len(series["data"]) - len(_objcnt)
                expand_obj += [""] * delta
                expand_ref += [""] * delta

            for data, title, ref_len in zip(series["data"], expand_obj, expand_ref):
                data["title"] = title
                data["referencesCount"] = ref_len

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
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return
        _data = dict(self._objects_cnt_dict)
        _refs = dict(self._references_dict)
        for tag_id, vals in dict(_data).items():
            _data[tag_id] = dict(vals)
            for val, obj_cnt in dict(vals).items():
                _data[tag_id][val] = [obj_cnt, _refs[tag_id][val]]

        return np.array(_data, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return
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
        if len(figures) == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return

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
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
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
            title="Objects",
            color_range="row",
            tooltip="Click to preview {y} objects from {referencesCount} images with tag {series_name} and value {tag_value}",
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
        _tags_objcnt = dict(self._objects_cnt_dict).values()
        _tags_refscnt = dict(self._references_dict).values()
        for series, _objcnt, _refscnt in zip(res["series"], _tags_objcnt, _tags_refscnt):
            expand_obj = list(_objcnt)
            expand_ref = [len(x) for x in _refscnt.values()]
            if len(_objcnt) < len(series["data"]):
                delta = len(series["data"]) - len(_objcnt)
                expand_obj += [""] * delta
                expand_ref += [""] * delta

            for data, title, ref_len in zip(series["data"], expand_obj, expand_ref):
                data["title"] = title
                data["referencesCount"] = ref_len

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
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return
        _data = dict(self._objects_cnt_dict)
        _refs = dict(self._references_dict)
        for tag_id, vals in dict(_data).items():
            _data[tag_id] = dict(vals)
            for val, obj_cnt in dict(vals).items():
                _data[tag_id][val] = [obj_cnt, _refs[tag_id][val]]

        return np.array(_data, dtype=object)

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str) -> np.ndarray:
        if self._num_tags == 0 or self._num_tags > MAX_NUMBER_OF_TAGS:
            return
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

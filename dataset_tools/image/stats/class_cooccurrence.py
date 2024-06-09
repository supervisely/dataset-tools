import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo

from dataset_tools.image.stats.basestats import BaseStats

REFERENCES_LIMIT = 500
OBJECTS_TAGS = ["objectsOnly", "all"]


class ClassCooccurrence(BaseStats):
    """
    Columns:
        Class
        class 1
        class 2
        etc.
    """

    def __init__(
        self, project_meta: sly.ProjectMeta, cls_prevs_tags: list = [], force: bool = False
    ) -> None:
        self._meta = project_meta
        self._stats = {}
        self._cls_prevs_tags = cls_prevs_tags
        self.force = force

        self._name_to_index = {}

        self._class_names = [cls.name for cls in project_meta.obj_classes] + self._cls_prevs_tags

        for idx, obj_class_name in enumerate(self._class_names):
            self._name_to_index[obj_class_name] = idx

        self._references = defaultdict(lambda: defaultdict(list))

        self._num_classes = len(self._class_names)
        self.co_occurrence_matrix = np.zeros(
            (self._num_classes, self._num_classes), dtype=int
        )  # TODO maybe rm numpy sewing at all?

        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes.items()}
        self._class_to_index = {}

        for idx, obj_class in enumerate(self._meta.obj_classes):
            self._class_to_index[obj_class.sly_id] = idx

        self._images_set = {class_id: set() for class_id in self._class_ids}

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            return
        if self._num_classes == 0:
            return

        classes = set()
        for f in figures:
            classes.add(f.class_id)

        for class_id in classes:
            idx = self._class_to_index[class_id]
            self.co_occurrence_matrix[idx][idx] += 1
            self._references[idx][idx].append(image.id)

        classes = list(classes)
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = self._class_to_index[classes[i]]
                idx_j = self._class_to_index[classes[j]]
                self.co_occurrence_matrix[idx_i][idx_j] += 1
                self.co_occurrence_matrix[idx_j][idx_i] += 1

                self._references[idx_i][idx_j].append(image.id)
                self._references[idx_j][idx_i].append(image.id)

    def to_json2(self):
        return self.to_json()

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        if self._num_classes == 1:
            return
        classes = set()
        for label in ann.labels:
            classes.add(label.obj_class.name)

        for tag in ann.img_tags:
            if tag.name in self._cls_prevs_tags:
                classes.add(tag.name)

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
        self.__init__(self._meta, self._cls_prevs_tags, self.force)

    def to_json(self) -> Optional[Dict]:
        if self._num_classes == 0:
            return
        # if self.co_occurrence_matrix is None:
        #     return None
        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. {currentCell} images have objects of both classes {firstCell} and {currentColumn} at same time",
        }
        colomns_options = [None] * (len(self._class_names) + 1)
        colomns_options[0] = {"type": "class"}  # not used in Web

        for idx in range(1, len(colomns_options)):
            colomns_options[idx] = {"maxValue": int(np.max(self.co_occurrence_matrix[:, idx - 1]))}

        data = [
            [value] + sublist
            for value, sublist in zip(self._class_names, self.co_occurrence_matrix.tolist())
        ]
        for i in range(len(self._references)):
            for j in range(len(self._references)):
                if len(self._references[i][j]) > REFERENCES_LIMIT:
                    self._seize_list_to_fixed_size(self._references[i][j], REFERENCES_LIMIT)

        res = {
            "columns": ["Class"] + self._class_names,
            "data": data,
            "referencesCell": self._references,  # row - col
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        if self._num_classes == 0:
            return
        #  if unlabeled
        # if np.sum(self.co_occurrence_matrix) == 0:
        #     return
        matrix = np.array(self.co_occurrence_matrix, dtype="int32")

        n = self._num_classes
        ref_list = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ref_list[i][j] = set(self._references[i][j])

        references = np.array(ref_list, dtype=object)
        return np.stack([matrix, references], axis=0)

    def sew_chunks(self, chunks_dir: str, updated_classes: List[str] = []) -> np.ndarray:
        if self._num_classes == 0:
            return
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        # res = None
        res = np.zeros((self._num_classes, self._num_classes), dtype=int)
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
                indices = list(sorted([self._class_names.index(cls) for cls in _updated_classes]))
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
            if loaded_data.shape[1] != self._num_classes:
                stat_data, ref_data = update_shape(loaded_data, updated_classes)

            # if res is None:
            #     res = np.zeros((self._num_classes, self._num_classes), dtype="int32")
            res = np.add(stat_data, res)

            if len(references) == 0:
                references = np.empty_like(res).tolist()

            references = [
                merge_elements(sublist1, sublist2)
                for sublist1, sublist2 in zip(references, ref_data.tolist())
            ]

            np.save(file, np.stack([stat_data, ref_data]))

        # if res is None:
        #     np.zeros((self._num_classes, self._num_classes), dtype=int)

        self.co_occurrence_matrix = res
        for i, sublist in enumerate(references):
            for j, inner_list in enumerate(sublist):
                self._references[i][j] = list(inner_list)

        return res


class ClassToTagCooccurrence(BaseStats):
    def __init__(self, project_meta: sly.ProjectMeta, force: bool = False) -> None:
        self._meta = project_meta
        self.force = force

        self._class_names = [cls.name for cls in project_meta.obj_classes]

        self._tag_names = [
            cls.name for cls in project_meta.tag_metas if cls.applicable_to in OBJECTS_TAGS
        ]

        self._name_to_index = {}

        for idx, obj_class_name in enumerate(self._class_names):
            self._name_to_index[obj_class_name] = idx

        self._tag_name_to_index = {}

        for idx, tag_name in enumerate(self._tag_names):
            self._tag_name_to_index[tag_name] = idx

        # self._references = defaultdict(lambda: defaultdict(set))

        self._num_classes = len(self._class_names)
        self._num_tags = len(self._tag_names)
        # self.co_occurrence_matrix = np.zeros((self._num_classes, self._num_tags), dtype=int)

        # new
        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes}
        self._tag_ids = {
            item.sly_id: item.name for item in self._meta.tag_metas if item.name in self._tag_names
        }
        self.co_occurrence_dict = {
            class_id_x: {tag_id_y: set() for tag_id_y in self._tag_ids}
            for class_id_x in self._class_ids
        }
        self.references_dict = {
            class_id_x: {tag_id_y: set() for tag_id_y in self._tag_ids}
            for class_id_x in self._class_ids
        }

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            return
        if self._num_classes == 0:
            return

        for figure in figures:
            tags = set()
            for tag in figure.tags:
                tags.add(tag["tagId"])

            for tag_id in tags:
                self.co_occurrence_dict[figure.class_id][tag_id].add(figure.id)
                self.references_dict[figure.class_id][tag_id].add(image.id)

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        for label in ann.labels:
            for tag in label.tags:
                tag_name = tag.name

                idx_i = self._name_to_index[label.obj_class.name]
                idx_j = self._tag_name_to_index[tag_name]

                self.co_occurrence_matrix[idx_i][idx_j] += 1
                if len(self._references[idx_i][idx_j]) <= REFERENCES_LIMIT:
                    self._references[idx_i][idx_j].add(image.id)

    def clean(self) -> None:
        self.__init__(self._meta, self.force)

    def to_json2(self) -> Optional[Dict]:
        if self._num_classes == 0:
            return
        if self._num_tags == 0:
            return

        options = {
            "fixColumns": 1,  # not used in Web
            "cellTooltip": "Click to preview. Class {RowCell} has {currentCell} objects of tag {currentColumn}",
        }
        colomns_options = [None] * (len(self._tag_names) + 1)
        colomns_options[0] = {"type": "tag"}  # not used in Web

        row_index = {key: idx for idx, key in enumerate(self._class_ids)}
        column_index = {key: idx for idx, key in enumerate(self._tag_ids)}
        nested_objects = [[None] * self._num_tags for _ in range(self._num_classes)]
        nested_images = [[None] * self._num_tags for _ in range(self._num_classes)]
        refs_image_ids = {
            x: {y: None for y in range(self._num_tags)} for x in range(self._num_classes)
        }

        for row_key, subdict in self.co_occurrence_dict.items():
            for col_key, object_ids in subdict.items():
                row_idx = row_index[row_key]
                col_idx = column_index[col_key]
                nested_objects[row_idx][col_idx] = len(object_ids)

        for row_key, subdict in self.references_dict.items():
            for col_key, image_ids in subdict.items():
                row_idx = row_index[row_key]
                col_idx = column_index[col_key]
                nested_images[row_idx][col_idx] = len(image_ids)
                refs_image_ids[row_idx][col_idx] = list(image_ids)

        data = [[value] + sublist for value, sublist in zip(self._class_names, nested_objects)]
        data_refs = [[value] + sublist for value, sublist in zip(self._tag_names, nested_images)]

        res = {
            "columns": ["Tag"] + list(self._tag_names),
            "rows": self._class_names,
            "data": data,
            "referencesCellCount": data_refs,
            "referencesCell": refs_image_ids,
            "options": options,
            "colomnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        if self._num_classes == 0:
            return
        if self._num_tags == 0:
            return
        return np.array(
            {"objects": self.co_occurrence_dict, "images": self.references_dict}, dtype=object
        )

    def sew_chunks(self, chunks_dir: str, updated_classes: List[str] = []) -> np.ndarray:
        if self._num_classes == 0:
            return
        if self._num_tags == 0:
            return

        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                loaded_classes = set(loaded_data["objects"])
                true_classes = set(self._class_ids)
                fkey = next(iter(loaded_data["objects"]))
                loaded_tags = set(loaded_data["objects"][fkey])
                true_tags = set(self._tag_ids)

                added_cls = true_classes - loaded_classes
                added_tgs = true_tags - loaded_tags

                removed_cls = loaded_classes - true_classes
                removed_tgs = loaded_tags - true_tags

                for data, res in zip(
                    (loaded_data["objects"], loaded_data["images"]),
                    (self.co_occurrence_dict, self.references_dict),
                ):

                    for cls_id_new in added_cls:
                        data[cls_id_new] = dict()
                        for tag_id in true_tags:
                            data[cls_id_new][tag_id] = set()
                    for cls_id in true_classes:
                        for tag_id_new in added_tgs:
                            data[cls_id][tag_id_new] = set()

                    for cls_id_rm in removed_cls:
                        data.pop(cls_id_rm)
                    for cls_id in true_classes:
                        for tag_id_rm in removed_tgs:
                            data[cls_id].pop(tag_id_rm)

                    for cls_id in true_classes:
                        for tag_id in true_tags:
                            res[cls_id][tag_id].update(data[cls_id][tag_id])

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

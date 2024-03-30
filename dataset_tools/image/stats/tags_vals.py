import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]
TAGSS_CNT_LIMIT = 100

MAX_SIZE_OBJECT_SIZES_BYTES = 1e7
SHRINKAGE_COEF = 0.1
REFERENCES_LIMIT = 1000


class TagsVals(BaseStats):
    """
    Columns:
        Image
        Dataset
        Height
        Width
        Unlabeled
        Class1 objects count
        Class1 covered area (%)
        Class2 objects count
        Class2 covered area (%)
        etc.
    """

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        # project_stats: dict,
        datasets: List[sly.DatasetInfo] = None,
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        # self.project_stats = project_stats
        self.datasets = datasets
        self.force = force
        self._stat_cache = stat_cache

        self._stats = {}

        self._dataset_id_to_name = None
        if datasets is not None:
            self._dataset_id_to_name = {ds.id: ds.name for ds in datasets}

        self._tags_vals_count = defaultdict(lambda: defaultdict(int))
        self._tag_name_to_rows = {}
        self._columns = ["Tag"]
        self._stats["data"] = []

        self._temp_references = defaultdict(lambda: defaultdict(list))

        self._references = defaultdict(lambda: defaultdict(list))

    def clean(self):
        self.__init__(
            self._meta,
            self.project_stats,
            self.datasets,
            self.force,
            self._stat_cache,
        )

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

        for tag in ann.img_tags:
            if tag.value is None:
                continue

            self._tags_vals_count[tag.name][tag.value] += 1
            self._temp_references[tag.name][tag.value].append(image.id)

    def to_json(self) -> Dict:
        columns = ["Tag"]
        columns_options = [None] * len(columns)

        already_added_data_len = 0

        for idx, tag_name in enumerate(self._tags_vals_count.keys()):
            table_row = [tag_name]
            zero_idx = 0
            for zero_idx, i in enumerate(range(already_added_data_len)):
                table_row.append(0)
            for val_idx, tag_value in enumerate(self._tags_vals_count[tag_name]):
                columns.append(tag_value)
                table_row.append(self._tags_vals_count[tag_name][tag_value])
                image_ids_list = self._temp_references[tag_name][tag_value]
                self._references[idx + 1][zero_idx + val_idx + 1].append(image_ids_list)
                already_added_data_len += 1
                columns_options.append({"subtitle": "tags count"})

            self._stats["data"].append(table_row)

        options = {"fixColumns": 1}
        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": self._stats["data"],
            "options": options,
            "referencesCell": self._references,
        }
        return res

    def to_numpy_raw(self):
        return np.array(
            [[a] + [b] for a, b in zip(self._stats["data"], self._references)],
            dtype=object,
        )

    def sew_chunks(self, chunks_dir: str, updated_tags: List[str] = []):
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        res = []
        is_zero_area = None
        references = []
        labeled_cls = self._class_names[1:]

        def update_shape(loaded_data: list, updated_tags, insert_val=0) -> list:
            if len(updated_tags) > 0:
                indices = list(sorted([labeled_cls.index(cls) for cls in updated_tags]))
                for idx, image in enumerate(loaded_data):
                    stat_data, ref_data = image
                    cls_data = stat_data[5:]
                    for ind in indices:
                        cls_data.insert(2 * ind, insert_val)
                        cls_data.insert(2 * ind + 1, insert_val)
                    stat_data = stat_data[:5] + cls_data
                    loaded_data[idx] = [stat_data, ref_data]
            return loaded_data

        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if len(loaded_data[0][0][5:]) != (len(labeled_cls) * 2):
                loaded_data = update_shape(loaded_data, updated_tags)

            for image in loaded_data:
                stat_data, ref_data = image
                res.append(stat_data)
                references.append(ref_data)

            save_data = np.array(loaded_data, dtype=object)
            np.save(file, save_data)

        self._stats["data"] = res
        self._references = references

        return np.array(res)

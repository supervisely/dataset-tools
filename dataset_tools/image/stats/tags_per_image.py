import random
from typing import Dict, List

import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]
TAGS_CNT_LIMIT = 100

MAX_SIZE_OBJECT_SIZES_BYTES = 1e7
SHRINKAGE_COEF = 0.1


class TagsPerImage(BaseStats):
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
        project_stats: dict,
        datasets: List[sly.DatasetInfo] = None,
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        self.project_stats = project_stats
        self.datasets = datasets
        self.force = force
        self._stat_cache = stat_cache

        self._stats = {}

        self._dataset_id_to_name = None
        if datasets is not None:
            self._dataset_id_to_name = {ds.id: ds.name for ds in datasets}

        self._tag_names = []
        # self._class_indices_colors = [UNLABELED_COLOR]
        self._tagname_to_index = {}

        for idx, tag_meta in enumerate(self._meta.tag_metas):
            if idx >= TAGS_CNT_LIMIT:
                sly.logger.warn(f"{self.__class__.__name__}: will use first {TAGS_CNT_LIMIT} tags.")
                break
            self._tag_names.append(tag_meta.name)
            tag_index = idx + 1
            # self._class_indices_colors.append([class_index, class_index, class_index])
            self._tagname_to_index[tag_meta.name] = tag_index

        self._stats["data"] = []
        self._referencesRow = []

        # total = self.project_stats["images"]["total"]["imagesInDataset"] * (
        #     len(self.project_stats["images"]["objectClasses"]) + 5
        # )
        # self.update_freq = 1
        # if total > MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF:
        #     self.update_freq = MAX_SIZE_OBJECT_SIZES_BYTES * SHRINKAGE_COEF / total

    def clean(self):
        self.__init__(
            self._meta,
            self.project_stats,
            self.datasets,
            self.force,
            self._stat_cache,
        )

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:

        table_row = []

        table_row.append(image.name)

        if self._dataset_id_to_name is not None:
            table_row.append(self._dataset_id_to_name[image.dataset_id])

        table_row.extend(
            [
                image.height,  # stat_area["height"],
                image.width,  # stat_area["width"],
            ]
        )

        table_row.extend([None] * len(self._tag_names))

        for tag in ann.img_tags:
            tag_index = self._tagname_to_index[tag.name] + 3
            tag_value = tag.value
            if tag_value is None:
                tag_value = "none"
            table_row[tag_index] = tag_value

        self._stats["data"].append(table_row)
        self._referencesRow.append([image.id])

    def to_json(self) -> Dict:
        if self._dataset_id_to_name is not None:
            columns = ["Image", "Split", "Height", "Width"]
        else:
            columns = ["Image", "Height", "Width"]

        columns_options = [None] * len(columns)

        if self._dataset_id_to_name is not None:
            columns_options[columns.index("Split")] = {
                "subtitle": "folder name",
            }
        columns_options[columns.index("Height")] = {
            "postfix": "px",
        }
        columns_options[columns.index("Width")] = {
            "postfix": "px",
        }

        for tag_name in self._tag_names:
            columns_options.append({"subtitle": "tag name"})
            columns.extend([tag_name])

        options = {"fixColumns": 1}
        res = {
            "columns": columns,
            "columnsOptions": columns_options,
            "data": self._stats["data"],
            "options": options,
            "referencesRow": self._referencesRow,
        }
        return res

    def to_numpy_raw(self):
        return np.array(
            [[a] + [b] for a, b in zip(self._stats["data"], self._referencesRow)],
            dtype=object,
        )

    def sew_chunks(self, chunks_dir: str, updated_classes: List[str] = []):
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])

        res = []
        is_zero_area = None
        references = []
        labeled_cls = self._class_names[1:]

        def update_shape(loaded_data: list, updated_classes, insert_val=0) -> list:
            if len(updated_classes) > 0:
                indices = list(sorted([labeled_cls.index(cls) for cls in updated_classes]))
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
                loaded_data = update_shape(loaded_data, updated_classes)

            for image in loaded_data:
                stat_data, ref_data = image
                res.append(stat_data)
                references.append(ref_data)

            save_data = np.array(loaded_data, dtype=object)
            np.save(file, save_data)

        self._stats["data"] = res
        self._referencesRow = references

        return np.array(res)

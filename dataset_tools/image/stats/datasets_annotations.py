from typing import Dict, List, Optional

import supervisely as sly
from dataset_tools.image.stats.basestats import BaseStats
import numpy as np
from collections import defaultdict

class DatasetsAnnotations(BaseStats):
    """
    Aggregated statistics for datasets annotations.

    This statistic builds a table using data from project_stats.
    """
    def __init__(self, project_meta: sly.ProjectMeta, project_stats: Dict, datasets: List, force: bool = False,
                 stat_cache: dict = None) -> None:
        self._meta = project_meta
        self._project_stats = project_stats
        self._datasets = datasets
        self.force = force
        self._stat_cache = stat_cache

        # get aggregated names to use as rows
        self._id_to_info = {ds.id : ds for ds in datasets}
        self._id_to_name = {}
        self._get_aggregated_names(datasets)
        self._class_id_to_name = {cls.sly_id: cls.name for cls in project_meta.obj_classes}

        # mappings for parent-child relationships
        self._id_to_parents = defaultdict(list)
        for ds in datasets:
            current = ds
            while current.parent_id:
                current = self._id_to_info[current.parent_id]
                self._id_to_parents[ds.id].append(current)

        self._parent_to_infos = defaultdict(list)
        for ds in datasets:
            current = ds
            while parent_id := current.parent_id:
                parent_id = current.parent_id
                self._parent_to_infos[parent_id].append(ds)
                current = self._id_to_info[parent_id]

        # get aggregated total number of images
        self._id_to_total = {}
        for ds in datasets:
            self._id_to_total[ds.id] = ds.images_count
            for children in self._parent_to_infos.get(ds.id, []):
                self._id_to_total[ds.id] += children.images_count

        # annotations statistics
        self._num_objects = {ds.id: 0 for ds in datasets}
        self._num_tagged_objs = {ds.id: 0 for ds in datasets}
        self._num_annotated = {ds.id: 0 for ds in datasets}
        self._num_tagged = {ds.id: 0 for ds in datasets}

        self._class_avg_cnt = defaultdict(dict)
        self._class_avg_area = defaultdict(dict)
        for ds_id in self._id_to_info.keys():
            for cls in project_meta.obj_classes:
                self._class_avg_cnt[ds_id][cls.sly_id] = 0
                self._class_avg_area[ds_id][cls.sly_id] = 0

        self._class_cnt_sum = defaultdict(dict)
        self._class_area_sum = defaultdict(dict)

        # images for references
        self._images_set = defaultdict(set)
        for ds_id in self._id_to_info.keys():
            for cls in project_meta.obj_classes:
                self._class_cnt_sum[ds_id][cls.sly_id] = 0
                self._class_area_sum[ds_id][cls.sly_id] = 0

        self.is_unlabeled = True

    def clean(self) -> None:
        self.__init__(
            self._meta,
            self._project_stats,
            self._datasets,
            self.force,
            self._stat_cache,
        )

    def _get_aggregated_names(self, datasets: List) -> Dict:
        for dataset in datasets:
            original_id = dataset.id
            dataset_name = dataset.name
            current = dataset
            while parent := current.parent_id:
                dataset_name = self._id_to_info[parent].name + '/' + dataset_name
                current = self._id_to_info[parent]
            self._id_to_name[original_id] = dataset_name

    def update(self):
        raise NotImplementedError()

    def update2(self, image, figures) -> None:
        ds_id = image.dataset_id
        parents = self._id_to_parents.get(ds_id, [])
        if len(image.tags) > 0:
            self._num_tagged[ds_id] += 1
            for parent in parents:
                self._num_tagged[parent.id] += 1

        if len(figures) == 0:
            return
        self._num_annotated[ds_id] += 1 if len(figures) > 0 else 0
        self._num_objects[ds_id] += len(figures)
        self._num_tagged_objs[ds_id] += len([figure for figure in figures if figure.tags])
        for parent in parents:
            self._num_annotated[parent.id] += 1
            self._num_objects[parent.id] += len(figures)
            self._num_tagged_objs[parent.id] += len([figure for figure in figures if figure.tags])
        self.is_unlabeled = False

        image_area = image.width * image.height
        for figure in figures:
            self._images_set[ds_id].add(image.id)
            self._class_cnt_sum[ds_id][figure.class_id] += 1
            self._class_area_sum[ds_id][figure.class_id] += float(figure.area) / image_area * 100
            for parent in parents:
                self._images_set[parent.id].add(image.id)
                self._class_cnt_sum[parent.id][figure.class_id] += 1
                self._class_area_sum[parent.id][figure.class_id] += float(figure.area) / image_area * 100

    def to_json(self):
        raise NotImplementedError()

    def to_json2(self) -> Optional[Dict]:
        columns = ["Dataset", "ID", "Total", "Annotated", "Tagged", "Objects", "Tagged objects"]
        for name in self._class_id_to_name.values():
            columns.append(name)
            columns.append(name)

        # max_images = max(list(self._id_to_total.values()))
        # max_annotated = max(list(self._num_annotated.values()))
        # max_tagged = max(list(self._num_tagged.values()))

        col_options = [None] * len(columns)
        col_options[0] = {}
        col_options[1] = {}
        col_options[2] = {"subtitle": "number of images"}
        col_options[3] = {"subtitle": "number of images"}
        col_options[4] = {"subtitle": "images count"}
        col_options[5] = {"subtitle": "total count"}
        col_options[6] = {"subtitle": "total count"}

        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "asc"},
            "pageSize": 10,
        }

        count_tooltip = "Average count of objects per image"
        area_tooltip = "Average area covered by objects per image"
        rows, refs = [], []
        for ds_id, ds_name in self._id_to_name.items():
            total = self._id_to_total[ds_id]
            num_ann = self._num_annotated[ds_id]
            num_tag = self._num_tagged[ds_id]
            num_obj = self._num_objects[ds_id]
            num_tagged_objs = self._num_tagged_objs[ds_id]
            row = [ds_name, ds_id, total, num_ann, num_tag, num_obj, num_tagged_objs]
            for idx, class_id in enumerate(self._class_id_to_name.keys()):
                self._class_avg_cnt[ds_id][class_id] = (
                    round(self._class_cnt_sum[ds_id][class_id] / num_ann, 2) if num_ann else 0
                )
                self._class_avg_area[ds_id][class_id] = (
                    round(self._class_area_sum[ds_id][class_id] / num_ann, 2) if num_ann else 0
                )
                max_value_per_class_cnt = max([cnt for class_to_cnt in self._class_avg_cnt.values() for clss, cnt in class_to_cnt.items() if clss == class_id])
                # max_value_per_class_area = max([area for class_to_area in self._class_avg_area.values() for clss, area in class_to_area.items() if clss == class_id])
                col_options[7 + 2 * idx] = {"maxValue": max_value_per_class_cnt, "subtitle": "average number of objects per image", "tooltip": count_tooltip}
                col_options[8 + 2 * idx] = {
                    "maxValue": 100,
                    "subtitle": "average covered area",
                    "tooltip": area_tooltip,
                    "postfix": "%",
                }

                row.append(self._class_avg_cnt[ds_id][class_id])
                row.append(self._class_avg_area[ds_id][class_id])
            rows.append(row)

            seized_refs = self._seize_list_to_fixed_size(list(self._images_set[ds_id]), 1000)
            refs.append(seized_refs)

        return {
            "columns": columns,
            "data": rows,
            "referencesRow": refs,
            "options": options,
            "columnsOptions": col_options,
        }

    def to_numpy_raw(self):
        if self.is_unlabeled:
            return
        images_set = np.array(self._images_set, dtype=object)
        avg_class_cnt_sum = np.array(self._class_cnt_sum, dtype=object)
        avg_class_area_sum = np.array(self._class_area_sum, dtype=object)
        num_annotated = np.array(self._num_annotated, dtype=object)
        num_tagged = np.array(self._num_tagged, dtype=object)
        num_objects = np.array(self._num_objects, dtype=object)
        num_tagged_objs = np.array(self._num_tagged_objs, dtype=object)
        return np.stack(
            [
                images_set,
                avg_class_cnt_sum,
                avg_class_area_sum,
                num_annotated,
                num_tagged,
                num_objects,
                num_tagged_objs
            ],
            axis=0,
        )

    def sew_chunks(self, chunks_dir: str) -> None:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                loaded_ds_ids = set(loaded_data[0].keys())
                true_ds_ids = set(self._id_to_name.keys())

                added = true_ds_ids.difference(loaded_ds_ids)
                for ds_id in added:
                    loaded_data[0][ds_id] = set()
                    loaded_data[1][ds_id] = {class_id: 0 for class_id in self._class_id_to_name.keys()}
                    loaded_data[2][ds_id] = {class_id: 0 for class_id in self._class_id_to_name.keys()}
                    loaded_data[3][ds_id] = 0
                    loaded_data[4][ds_id] = 0
                    loaded_data[5][ds_id] = 0
                    loaded_data[6][ds_id] = 0

                removed = loaded_ds_ids.difference(true_ds_ids)
                for ds_id in list(removed):
                    loaded_data[0].pop(ds_id)
                    loaded_data[1].pop(ds_id)
                    loaded_data[2].pop(ds_id)
                    loaded_data[3].pop(ds_id)
                    loaded_data[4].pop(ds_id)
                    loaded_data[5].pop(ds_id)
                    loaded_data[6].pop(ds_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for ds_id in loaded_ds_ids:
                    self._images_set[ds_id].update(loaded_data[0][ds_id])
                    for class_id, cnt in loaded_data[1][ds_id].items():
                        self._class_cnt_sum[ds_id][class_id] += cnt
                    for class_id, area in loaded_data[2][ds_id].items():
                        self._class_area_sum[ds_id][class_id] += area
                    self._num_annotated[ds_id] += loaded_data[3][ds_id]
                    self._num_tagged[ds_id] += loaded_data[4][ds_id]
                    self._num_objects[ds_id] += loaded_data[5][ds_id]
                    self._num_tagged_objs[ds_id] += loaded_data[6][ds_id]

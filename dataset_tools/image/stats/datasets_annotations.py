from typing import Dict, List, Optional

import supervisely as sly
from dataset_tools.image.stats.basestats import BaseStats
import numpy as np

class DatasetsAnnotations(BaseStats):
    """
    Statistics for datasets annotations.

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
        self._id_to_name = {}
        self._id_to_info = {ds.id : ds for ds in datasets}
        self._get_aggregated_names(datasets)

        self._id_to_total = {ds_id: info.images_count for ds_id, info in self._id_to_info.items()}
        self._class_id_to_name = {cls.sly_id: cls.name for cls in project_meta.obj_classes}

        self._num_annotated = {ds_id: 0 for ds_id in self._id_to_info.keys()}
        self._num_tagged = {ds_id: 0 for ds_id in self._id_to_info.keys()}
        self._class_avg_cnt = {}
        self._class_avg_area = {}

        self._class_cnt_sum = {}
        self._class_area_sum = {}

        self._images_set = {}
        for ds_id in self._id_to_info.keys():
            self._images_set[ds_id] = set()
            self._class_cnt_sum[ds_id] = {}
            self._class_area_sum[ds_id] = {}
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
            while True:
                parent = current.parent_id
                if parent is None:
                    break
                current = self._id_to_info[parent]
                dataset_name = current.name + '/' + dataset_name
            self._id_to_name[original_id] = dataset_name

    def update(self):
        raise NotImplementedError()
    
    def update2(self, image, figures) -> None:
        ds_id = image.dataset_id
        if len(image.tags) > 0:
            self._num_tagged[ds_id] += 1
        if len(figures) == 0:
            return
        self.is_unlabeled = False

        self._num_annotated[ds_id] += 1
        for figure in figures:
            self._images_set[ds_id].add(image.id)
            self._class_cnt_sum[ds_id][figure.class_id] += 1
            self._class_area_sum[ds_id][figure.class_id] += int(figure.area)

    def to_json(self):
        raise NotImplementedError()
    
    def to_json2(self) -> Optional[Dict]:
        columns = ["Dataset", "ID", "Total Images Count", "Annotated Images Count", "Tagged Images Count"]
        for name in self._class_id_to_name.values():
            columns.append(f"{name} Avg Count")
            columns.append(f"{name} Avg Area")

        max_images = max(list(self._id_to_total.values()))
        max_annotated = max(list(self._num_annotated.values()))
        max_tagged = max(list(self._num_tagged.values()))

        col_options = [None] * len(columns)
        col_options[0] = {"type": "dataset"}
        col_options[1] = {"type": "id"}
        col_options[2] = {"maxValue": max_images, "tooltip": "Total number of images in the dataset"}
        col_options[3] = {"maxValue": max_annotated, "tooltip": "Total number of annotated images"}
        col_options[4] = {"maxValue": max_tagged, "tooltip": "Total number of tagged images"}

        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "asc"},
            "pageSize": 10,
        }

        count_tooltip = "Average count of objects per image"
        area_tooltip = "Average area of objects per image"
        rows, refs = [], []
        for ds_id, ds_name in self._id_to_name.items():
            for idx, class_id in enumerate(self._class_cnt_sum[ds_id]):
                col_options[5 + 2 * idx] = {"maxValue": max(list(self._class_cnt_sum[ds_id].values())), "subtitle": "average", "tooltip": count_tooltip}
                col_options[6 + 2 * idx] = {"maxValue": max(list(self._class_area_sum[ds_id].values())), "subtitle": "average", "tooltip": area_tooltip,"postfix": "%"}
                try:
                    self._class_avg_cnt[class_id] = self._class_cnt_sum[ds_id][class_id] / self._num_annotated[ds_id]
                except ZeroDivisionError:
                    self._class_avg_cnt[class_id] = 0
                try:
                    self._class_avg_area[class_id] = self._class_area_sum[ds_id][class_id] / self._num_annotated[ds_id]
                except ZeroDivisionError:
                    self._class_avg_area[class_id] = 0

            total = self._id_to_total[ds_id]
            num_ann = self._num_annotated[ds_id]
            num_tag = self._num_tagged[ds_id]
            row = [ds_name, ds_id, total, num_ann, num_tag]
            for class_id in self._class_id_to_name.keys():
                row.append(self._class_avg_cnt[class_id])
                row.append(self._class_avg_area[class_id])
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
        return np.stack(
            [
                images_set,
                avg_class_cnt_sum,
                avg_class_area_sum,
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

                removed = loaded_ds_ids.difference(true_ds_ids)
                for ds_id in list(removed):
                    loaded_data[0].pop(ds_id)
                    loaded_data[1].pop(ds_id)
                    loaded_data[2].pop(ds_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for ds_id in self._id_to_name.keys():
                    self._images_set[ds_id].update(loaded_data[0][ds_id])
                    self._class_cnt_sum[ds_id] = loaded_data[1][ds_id]
                    self._class_area_sum[ds_id] = loaded_data[2][ds_id]
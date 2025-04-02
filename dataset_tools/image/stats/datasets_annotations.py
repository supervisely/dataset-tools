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
        self._id_to_name = self._get_aggregated_names(datasets, self._id_to_info)
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

        self._class_figure_areas = {ds.id: defaultdict(list) for ds in datasets}
        self._class_imgids = {ds.id: defaultdict(set) for ds in datasets}
        self._imgid_to_area = {ds.id: {} for ds in datasets}

        self._images_set = defaultdict(set)

        self.is_unlabeled = True

    def clean(self) -> None:
        self.__init__(
            self._meta,
            self._project_stats,
            self._datasets,
            self.force,
            self._stat_cache,
        )

    def _get_aggregated_names(self, datasets: List, id_to_info: Dict) -> Dict:
        id_to_name = {}
        for dataset in datasets:
            original_id = dataset.id
            dataset_name = dataset.name
            current = dataset
            while parent := current.parent_id:
                dataset_name = id_to_info[parent].name + '/' + dataset_name
                current = id_to_info[parent]
            id_to_name[original_id] = dataset_name
        return id_to_name

    def update(self):
        raise NotImplementedError()

    def update2(self, image, figures) -> None:
        parents = self._id_to_parents.get(image.dataset_id, [])
        ids_to_update = [image.dataset_id] + [parent.id for parent in parents]
        for i in ids_to_update:
            self._images_set[i].add(image.id)
            if len(image.tags) > 0:
                self._num_tagged[i] += 1

        if len(figures) == 0:
            return

        self.is_unlabeled = False
        for i in ids_to_update:
            self._imgid_to_area[i][image.id] = image.width * image.height
            self._num_annotated[i] += 1
            for figure in figures:
                self._class_imgids[i][figure.class_id].add(image.id)
                self._class_figure_areas[i][figure.class_id].append(int(figure.area))
                self._num_objects[i] += 1
                self._num_tagged_objs[i] += 1 if figure.tags else 0

    def to_json(self):
        raise NotImplementedError()

    def to_json2(self) -> Optional[Dict]:
        columns = ["Dataset", "ID", "Size", "Annotated", "Tagged", "Objects", "Tagged Objects"]
        for name in self._class_id_to_name.values():
            columns.append(name)
            columns.append(name)

        col_options = [None] * len(columns)
        col_options[0] = {}
        col_options[1] = {}
        col_options[2] = {"subtitle": "images count"}
        col_options[3] = {"subtitle": "images count"}
        col_options[4] = {"subtitle": "images count"}
        col_options[5] = {"subtitle": "total count"}
        col_options[6] = {"subtitle": "total count"}

        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "asc"},
            "pageSize": 10,
        }

        count_tooltip = (
            "Average count of objects per image. Only images with objects of class are counted."
        )
        area_tooltip = "Average area covered by objects per image. Only images with objects of class are taken into account."
        rows, refs = [], []
        # create mappings for class_id to average count and area
        class_avg_cnt = defaultdict(lambda: defaultdict(lambda: 0))
        class_areas = defaultdict(lambda: defaultdict(lambda: 0))
        for ds_id in self._id_to_total:
            for class_id in self._class_id_to_name:
                imgids = self._class_imgids[ds_id][class_id]

                obj_cnt = len(self._class_figure_areas[ds_id][class_id])
                img_cnt = len(imgids)
                avg_count = round(obj_cnt / img_cnt, 2) if img_cnt > 0 else 0
                class_avg_cnt[ds_id][class_id] = avg_count
                if img_cnt > 0:
                    class_area = sum(self._class_figure_areas[ds_id][class_id])
                    imgs_area = sum([self._imgid_to_area[ds_id][imgid] for imgid in imgids])
                    area = round((class_area / imgs_area) * 100, 2) if imgs_area > 0 else 0
                else:
                    area = 0
                class_areas[ds_id][class_id] = area

        for ds_id, ds_name in self._id_to_name.items():
            total = self._id_to_total[ds_id]
            num_ann = self._num_annotated[ds_id]
            num_tag = self._num_tagged[ds_id]
            num_obj = self._num_objects[ds_id]
            num_tagged_objs = self._num_tagged_objs[ds_id]
            row = [ds_name, ds_id, total, num_ann, num_tag, num_obj, num_tagged_objs]
            for idx, class_id in enumerate(self._class_id_to_name):
                class_cnt_avg = class_avg_cnt[ds_id][class_id]
                class_area_avg = class_areas[ds_id][class_id]
                row.append(class_cnt_avg)
                row.append(class_area_avg)

                col_options[7 + 2 * idx] = {"maxValue": max(class_avg_cnt[ds_id].values()), "subtitle": "objects per image", "tooltip": count_tooltip}
                col_options[8 + 2 * idx] = {
                    "maxValue": 100,
                    "subtitle": "average area",
                    "tooltip": area_tooltip,
                    "postfix": "%",
                }
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
        num_annotated = np.array(self._num_annotated, dtype=object)
        num_tagged = np.array(self._num_tagged, dtype=object)
        num_objects = np.array(self._num_objects, dtype=object)
        num_tagged_objs = np.array(self._num_tagged_objs, dtype=object)
        class_imgs = np.array(self._class_imgids, dtype=object)
        class_figure_areas = np.array(self._class_figure_areas, dtype=object)
        imgid_to_area = np.array(self._imgid_to_area, dtype=object)
        return np.stack(
            [
                images_set,
                num_annotated,
                num_tagged,
                num_objects,
                num_tagged_objs,
                class_imgs,
                class_figure_areas,
                imgid_to_area,
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
                    loaded_data[1][ds_id] = 0
                    loaded_data[2][ds_id] = 0
                    loaded_data[3][ds_id] = 0
                    loaded_data[4][ds_id] = 0
                    loaded_data[5][ds_id] = {
                        class_id: set() for class_id in self._class_id_to_name.keys()
                    }
                    loaded_data[6][ds_id] = {
                        class_id: [] for class_id in self._class_id_to_name.keys()
                    }
                    loaded_data[7][ds_id] = {}

                removed = loaded_ds_ids.difference(true_ds_ids)
                for ds_id in list(removed):
                    for i in range(len(loaded_data)):
                        loaded_data[i].pop(ds_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for ds_id in loaded_ds_ids:
                    self._images_set[ds_id].update(loaded_data[0][ds_id])
                    self._num_annotated[ds_id] += loaded_data[1][ds_id]
                    self._num_tagged[ds_id] += loaded_data[2][ds_id]
                    self._num_objects[ds_id] += loaded_data[3][ds_id]
                    self._num_tagged_objs[ds_id] += loaded_data[4][ds_id]
                    for class_id, areas in loaded_data[5][ds_id].items():
                        self._class_imgids[ds_id][class_id].update(areas)
                    for class_id, areas in loaded_data[6][ds_id].items():
                        self._class_figure_areas[ds_id][class_id].extend(areas)
                    self._imgid_to_area[ds_id].update(loaded_data[7][ds_id])

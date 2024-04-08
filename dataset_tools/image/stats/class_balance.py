import multiprocessing
import random
from typing import Dict, List, Optional, Tuple
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
import numpy as np
import supervisely as sly

from dataset_tools.image.stats.basestats import BaseStats

UNLABELED_COLOR = [0, 0, 0]
REFERENCES_LIMIT = 1000


class ClassBalance(BaseStats):
    """
    Columns:
        Class
        Images
        Objects
        Avg count per image
        Avg area per image
    """

    def __init__(
        self,
        project_meta: sly.ProjectMeta,
        project_stats,
        force: bool = False,
        stat_cache: dict = None,
    ) -> None:
        self._meta = project_meta
        self._project_stats = project_stats
        self.force = force

        # ! old
        self._stat_cache = stat_cache

        self._stats = {}
        self.references_probabilities = {}
        for cls in project_stats["images"]["objectClasses"]:
            self.references_probabilities[cls["objectClass"]["name"]] = (
                REFERENCES_LIMIT / cls["total"] if cls["total"] != 0 else 1
            )

        self.class_names = ["unlabeled"]
        class_colors = [UNLABELED_COLOR]
        class_indices_colors = [UNLABELED_COLOR]

        self._name_to_cls_id = {}  # new
        self._name_to_index = {}
        for idx, obj_class in enumerate(self._meta.obj_classes):
            self.class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            cls_idx = idx + 1
            class_indices_colors.append([cls_idx, cls_idx, cls_idx])
            self._name_to_index[obj_class.name] = cls_idx
            self._name_to_cls_id[obj_class.name] = cls_idx  # new

        self.class_indices_colors = class_indices_colors

        self.class_names = ["unlabeled"]
        self._class_names = []  # new
        for obj_class in self._meta.obj_classes:
            self.class_names.append(obj_class.name)
            self._class_names.append(obj_class.name)  # new
        # # TODO rm later
        self.sum_class_area_per_image = [0] * len(self.class_names)
        self.objects_count = [0] * len(self.class_names)
        self.images_count = [0] * len(self.class_names)

        self.image_counts_filter_by_id = [[] for _ in self.class_names]
        # self.dataset_counts_filter_by_id = [{} for _ in self.class_names]
        # self.ds_position = [0 for _ in self.class_names]
        # self.accum_ids = [set() for _ in self.class_names]

        self.avg_nonzero_area = [None] * len(self.class_names)
        self.avg_nonzero_count = [None] * len(self.class_names)

        # new
        self._class_ids = {item.sly_id: item.name for item in self._meta.obj_classes}

        self._images_set = {class_id: set() for class_id in self._class_ids}
        self._objects_set = {class_id: set() for class_id in self._class_ids}
        self._images_count = {class_id: 0 for class_id in self._class_ids}
        self._objects_count = {class_id: 0 for class_id in self._class_ids}
        self._count_on_image = {class_id: 0 for class_id in self._class_ids}
        self._area_figures_sum = {class_id: 0 for class_id in self._class_ids}
        self._area_images_sum = {class_id: 0 for class_id in self._class_ids}
        self._area_on_image_avg = {class_id: 0 for class_id in self._class_ids}
        self._area_images_percent_sum = {class_id: 0 for class_id in self._class_ids}

        self.is_unlabeled = True

    def clean(self) -> None:
        self.__init__(
            self._meta,
            self._project_stats,
            self.force,
            self._stat_cache,
        )

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):

        if len(figures) == 0:
            return
        self.is_unlabeled = False

        for figure in figures:
            self._images_set[figure.class_id].add(figure.entity_id)
            self._objects_set[figure.class_id].add(figure.id)
            self._area_images_percent_sum[figure.class_id] += int(figure.area) / (
                image.width * image.height
            )

    def to_json2(self) -> Optional[Dict]:
        for id in self._class_ids:
            self._images_count[id] = len(self._images_set[id])
            self._objects_count[id] = len(self._objects_set[id])
            try:
                self._count_on_image[id] = self._objects_count[id] / self._images_count[id]
                self._area_on_image_avg[id] = (
                    self._area_images_percent_sum[id] / self._images_count[id]
                )
            except ZeroDivisionError:
                self._count_on_image[id] = 0
                self._area_on_image_avg[id] = 0

        columns = [
            "Class",
            "Images",
            "Objects",
            "Count on image",
            "Area on image",
        ]
        rows, refs = [], []
        max_images_count, max_objects_count = 0, 0
        max_count_on_image, max_area_on_image = 0, 0
        for id, name in self._class_ids.items():
            count_on_image_avg = round(self._count_on_image[id], 2)
            area_on_image_avg = round(self._area_on_image_avg[id] * 100, 2)
            rows.append(
                [
                    name,
                    self._images_count[id],
                    self._objects_count[id],
                    count_on_image_avg,
                    area_on_image_avg,
                ]
            )
            seized_refs = self._seize_list_to_fixed_size(
                list(self._images_set[id]), REFERENCES_LIMIT
            )
            refs.append(seized_refs)
            max_images_count = max(max_images_count, self._images_count[id])
            max_objects_count = max(max_objects_count, self._objects_count[id])
            max_count_on_image = max(max_count_on_image, count_on_image_avg)
            max_area_on_image = max(max_area_on_image, area_on_image_avg)

        colomns_options = [None] * len(columns)
        colomns_options[0] = {"type": "class"}
        colomns_options[1] = {
            "maxValue": max_images_count,
            "tooltip": "Number of images with at least one object of corresponding class",
        }
        colomns_options[2] = {
            "maxValue": max_objects_count,
            "tooltip": "Number of objects of corresponding class in the project",
        }
        colomns_options[3] = {
            "maxValue": max_count_on_image,
            "subtitle": "average",
            "tooltip": "Average number of objects of corresponding class on the image. Images without such objects are not taking into account",
        }
        colomns_options[4] = {
            "postfix": "%",
            "maxValue": max_area_on_image,
            "subtitle": "average",
            "tooltip": "Average image area of corresponding class. Images without such objects are not taking into account",
        }
        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "desc"},
            "pageSize": 10,
        }  # asc

        res = {
            "columns": columns,
            "data": rows,
            "referencesRow": refs,
            "options": options,
            "columnsOptions": colomns_options,
        }
        return res

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        cur_class_names = ["unlabeled"]
        cur_class_colors = [UNLABELED_COLOR]
        classname_to_index = {}

        for label in ann.labels:
            if label.obj_class.name not in cur_class_names:
                cur_class_names.append(label.obj_class.name)
                class_index = len(cur_class_colors) + 1
                cur_class_colors.append([class_index, class_index, class_index])
                classname_to_index[label.obj_class.name] = class_index

        if self._stat_cache is not None and image.id in self._stat_cache:
            stat_area = self._stat_cache[image.id]["stat_area"]
        else:
            masks = []
            for cls in cur_class_names[1:]:
                render_rgb = np.zeros(ann.img_size + (3,), dtype="int32")

                class_labels = [label for label in ann.labels if label.obj_class.name == cls]
                clann = ann.clone(labels=class_labels)

                clann.draw(render_rgb, [1, 1, 1])
                masks.append(render_rgb)

            if len(masks) == 0:
                stat_area = {"unlabeled": 100}

            else:
                bitmasks1channel = [mask[:, :, 0] for mask in masks]
                stacked_masks = np.stack(bitmasks1channel, axis=2)
                total_area = stacked_masks.shape[0] * stacked_masks.shape[1]
                mask_areas = (np.sum(stacked_masks, axis=(0, 1)) / total_area) * 100

                mask_areas = np.insert(mask_areas, 0, self.calc_unlabeled_area_in(masks))
                stat_area = {cls: area for cls, area in zip(cur_class_names, mask_areas.tolist())}

                if self._stat_cache is not None:
                    if image.id in self._stat_cache:
                        self._stat_cache[image.id]["stat_area"] = stat_area
                    else:
                        self._stat_cache[image.id] = {"stat_area": stat_area}

        stat_count = ann.stat_class_count(cur_class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        for idx, class_name in enumerate(self.class_names):
            if class_name not in cur_class_names:
                cur_area = 0
                cur_count = 0
                self.images_count[idx] += 0
            else:
                cur_area = stat_area.get(
                    class_name, 0
                )  # if not np.isnan(stat_area[class_name]) else 0
                cur_count = stat_count.get(
                    class_name, 0
                )  # ] if not np.isnan(stat_count[class_name]) else 0
                self.images_count[idx] += 1 if cur_count > 0 else 0

            self.sum_class_area_per_image[idx] += cur_area
            self.objects_count[idx] += cur_count

            if self.images_count[idx] > 0:
                self.avg_nonzero_area[idx] = (
                    self.sum_class_area_per_image[idx] / self.images_count[idx]
                )
                self.avg_nonzero_count[idx] = self.objects_count[idx] / self.images_count[idx]

            if class_name in cur_class_names[1:]:
                if (
                    stat_count[class_name] > 0
                    and random.random() < self.references_probabilities[class_name]
                ):
                    self.image_counts_filter_by_id[idx].append(image.id)

                    # if image.dataset_id not in self.accum_ids[idx]:
                    #     self.dataset_counts_filter_by_id[idx].update(
                    #         {self.ds_position[idx]: image.dataset_id}
                    #     )
                    #     self.accum_ids[idx].add(image.dataset_id)
                    #     # self.accum_ids[idx] = list(set(self.accum_ids[idx]))
                    # self.ds_position[idx] += 1

    def to_json(self) -> Optional[Dict]:
        if len(self._meta.obj_classes) == 0:
            return None

        columns = [
            "Class",
            "Images",
            "Objects",
            "Count on image",
            "Area on image",
        ]
        rows = []

        for name, idx in self._name_to_cls_id.items():
            rows.append(
                [
                    name,
                    self.images_count[idx],
                    self.objects_count[idx],
                    round(self.avg_nonzero_count[idx] or 0, 2),
                    round(self.avg_nonzero_area[idx] or 0, 2),
                ]
            )
        notnonecount = [item for item in self.avg_nonzero_count if item is not None]
        notnonearea = [item for item in self.avg_nonzero_area if item is not None]

        colomns_options = [None] * len(columns)
        colomns_options[0] = {"type": "class"}
        colomns_options[1] = {
            "maxValue": max(self.images_count),
            "tooltip": "Number of images with at least one object of corresponding class",
        }
        colomns_options[2] = {
            "maxValue": max(self.objects_count),
            "tooltip": "Number of objects of corresponding class in the project",
        }
        colomns_options[3] = {
            "maxValue": round(max(notnonecount), 2),
            "subtitle": "average",
            "tooltip": "Average number of objects of corresponding class on the image. Images without such objects are not taking into account",
        }
        colomns_options[4] = {
            "postfix": "%",
            "maxValue": round(max(notnonearea), 2),
            "subtitle": "average",
            "tooltip": "Average image area of corresponding class. Images without such objects are not taking into account",
        }
        options = {
            "fixColumns": 1,
            "sort": {"columnIndex": 1, "order": "desc"},
            "pageSize": 10,
        }  # asc

        res = {
            "columns": columns,
            "data": rows,
            "referencesRow": self.image_counts_filter_by_id[1:],
            "options": options,
            "columnsOptions": colomns_options,
        }
        return res

    def to_numpy_raw(self):
        # if unlabeled
        if self.is_unlabeled:
            return
        # %%
        # images_count, objects_count = [], []
        # avg_cnt_on_img, sum_area_on_img = [], []
        # refs = []

        # for id in self._class_ids:
        #     objects_count.append(len(self._objects_set[id]))
        #     images_count.append(len(self._images_set[id]))
        #     try:
        #         avg_cnt_on_img.append(objects_count[-1] / images_count[-1])
        #         sum_area_on_img.append(
        #             self._area_images_percent_sum[id] / images_count[-1]
        #         )
        #     except ZeroDivisionError:
        #         avg_cnt_on_img.append(0)
        #         sum_area_on_img.append(0)

        #     refs.append(list(self._images_set[id]))

        # images_count = np.array(images_count, dtype="int32")
        # objects_count = np.array(objects_count, dtype="int32")
        # avg_cnt_on_img = np.array(
        #     [elem or 0 for elem in avg_cnt_on_img], dtype="float32"
        # )
        # sum_area_on_img = np.array(
        #     [elem or 0 for elem in sum_area_on_img], dtype="float32"
        # )
        # references = np.array(refs, dtype=object)

        # for figure in figures:
        #     self._images_set[figure.class_id].add(figure.entity_id)
        #     self._objects_set[figure.class_id].add(figure.id)
        #     self._area_images_percent_sum[figure.class_id] += int(figure.area) / (
        #         image.width * image.height
        #     )
        images_set = np.array(self._images_set, dtype=object)
        objects_set = np.array(self._objects_set, dtype=object)
        area_images_percent = np.array(self._area_images_percent_sum, dtype=object)
        return np.stack(
            [
                images_set,
                objects_set,
                area_images_percent,
            ],
            axis=0,
        )

    # @sly.timeit
    def sew_chunks(self, chunks_dir: str, updated_classes: dict = {}) -> np.ndarray:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                loaded_classes = set([class_id for class_id in loaded_data[0]])
                true_classes = set(self._class_ids)

                added = true_classes - loaded_classes
                for class_id in list(added):
                    loaded_data[0][class_id] = set()
                    loaded_data[1][class_id] = set()
                    loaded_data[2][class_id] = 0

                removed = loaded_classes - true_classes
                for class_id in list(removed):
                    loaded_data[0].pop(class_id)
                    loaded_data[1].pop(class_id)
                    loaded_data[2].pop(class_id)

                save_data = np.array(loaded_data, dtype=object)
                np.save(file, save_data)

                for class_id in self._class_ids:
                    self._images_set[class_id].update(loaded_data[0][class_id])
                    self._objects_set[class_id].update(loaded_data[1][class_id])
                    self._area_images_percent_sum[class_id] += loaded_data[2][class_id]

        return None

        # res = None
        # is_zero_area = None
        # references = None
        # none_chunks_cnt = 0

        # def update_shape(
        #     array: np.ndarray, updated_classes, insert_val=0
        # ) -> Tuple[np.ndarray, np.ndarray]:
        #     if len(updated_classes) > 0:
        #         indices = list(
        #             sorted([self.class_names.index(cls) for cls in updated_classes])
        #         )
        #         tmp = array.copy()
        #         for ind in indices:
        #             tmp = np.apply_along_axis(
        #                 lambda line: np.insert(line, [ind], [insert_val]),
        #                 axis=1,
        #                 arr=tmp,
        #             )
        #         sdata, rdata = tmp[:4, :], tmp[4, :]
        #         rdata = np.array(
        #             [[] if el == 0 else el for el in rdata.tolist()], dtype=object
        #         )
        #         return sdata, rdata
        #     return array[:4, :], array[4, :]

        # def concatenate_lists(a, b):
        #     return a + b if a and b else a if a else b

        # for file in files:
        #     loaded_data = np.load(file, allow_pickle=True)
        #     if np.any(loaded_data == None):
        #         none_chunks_cnt += 1
        #         continue

        #     stat_data, ref_data = loaded_data[:4, :], loaded_data[4, :]
        #     if loaded_data.shape[1] != len(self.class_names):
        #         stat_data, ref_data = update_shape(loaded_data, updated_classes)

        #     new_shape = (stat_data.shape[0], len(self.class_names))

        #     if references is None:
        #         references = [[] for _ in range(len(ref_data))]

        #     references = np.array(
        #         [concatenate_lists(a, b) for a, b in zip(ref_data, references)],
        #         dtype=object,
        #     )

        #     if res is None:
        #         res = np.zeros(new_shape)
        #     res = np.add(stat_data, res)

        #     if is_zero_area is None:
        #         is_zero_area = np.zeros(new_shape)[3]
        #     is_zero_area = np.add((stat_data[3] == 0).astype(int), is_zero_area)

        #     np.save(file, np.vstack([stat_data, ref_data]))

        # if none_chunks_cnt == len(files):
        #     sly.logger.warning(
        #         f"All chunks of {self.basename_stem} stat are None. Ignore sewing chunks."
        #     )
        #     return
        # # count on image
        # res[2] = res[1] / np.where(res[0] == 0, 1, res[0])

        # # area on image
        # area_denominators = np.array([len(files) - none_chunks_cnt] * new_shape[1])
        # area_denominators = area_denominators - is_zero_area
        # res[3] /= np.where(area_denominators == 0, 1, area_denominators)

        # cols = zip(
        #     list(self._class_ids),
        #     res[0].tolist(),
        #     res[1].tolist(),
        #     res[2].tolist(),
        #     res[3].tolist(),
        # )

        # for class_id, img, obj, cnt, area in cols:
        #     self._images_count[class_id] = img
        #     self._objects_count[class_id] = obj
        #     self._count_on_image[class_id] = cnt
        #     self._area_on_image_avg[class_id] = area

        # # self.images_count = res[0].tolist()
        # # self.objects_count = res[1].tolist()
        # # self.avg_nonzero_count = res[2].tolist()
        # # self.avg_nonzero_area = res[3].tolist()
        # # self.image_counts_filter_by_id = references.tolist()

        # return res

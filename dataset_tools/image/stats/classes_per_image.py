import numpy as np

import supervisely as sly


class ClassesPerImage:
    """
    Important fields of modified stats dict:
        "columns": [],
        "columnsOptions": [],
        "data": [],
        "table_options": dict
    """

    @property
    def dataset():
        pass

    @staticmethod
    def prepare_data(stats: dict, meta: sly.ProjectMeta):
        class_names = ["unlabeled"]
        class_indices_colors = [[0, 0, 0]]
        _name_to_index = {}
        table_columns = ["Image", "dataset", "height", "width", "channels", "unlabeled"]
        columns_options = ["null"] * len(table_columns)

        for idx, obj_class in enumerate(meta.obj_classes):
            class_names.append(obj_class.name)
            class_index = idx + 1
            class_indices_colors.append([class_index, class_index, class_index])
            _name_to_index[obj_class.name] = class_index
            columns_options.append({"subtitle": "objects count"})
            columns_options.append({"subtitle": "covered area (%)"})
            table_columns.extend([obj_class.name] * 2)


        # temp data
        stats["sum_class_area_per_image"] = [0] * len(class_names)
        stats["sum_class_count_per_image"] = [0] * len(class_names)
        stats["count_images_with_class"] = [0] * len(class_names)
        stats["class_names"] = class_names
        stats["class_indices_colors"] = class_indices_colors
        stats["_name_to_index"] = _name_to_index

        # important fields of stats
        stats["table_options"] = {"fixColumns": 1}
        stats["columns"] = table_columns
        stats["columnsOptions"] = columns_options
        stats["data"] = []

    @staticmethod
    def update(stats: dict, image_info, ann_info, meta, *args, **kwargs):
        BG_COLOR = [0, 0, 0]

        ann_json = ann_info.annotation
        ann = sly.Annotation.from_json(ann_json, meta)
        class_names = stats["class_names"]
        class_indices_colors = stats["class_indices_colors"]
        _name_to_index = stats["_name_to_index"]
        render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
        render_idx_rgb[:] = BG_COLOR
        ann.draw_class_idx_rgb(render_idx_rgb, _name_to_index)
        stat_area = sly.Annotation.stat_area(render_idx_rgb, class_names, class_indices_colors)
        stat_count = ann.stat_class_count(class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        table_row = []

        table_row.append(image_info.name)

        table_row.append(image_info.dataset_id)
        area_unl = stat_area["unlabeled"] if not np.isnan(stat_area["unlabeled"]) else 0
        table_row.extend(
            [
                stat_area["height"],
                stat_area["width"],
                stat_area["channels"],
                round(area_unl, 2),
            ]
        )
        for idx, class_name in enumerate(class_names):
            cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
            cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
            stats["sum_class_area_per_image"][idx] += cur_area
            stats["sum_class_count_per_image"][idx] += cur_count
            stats["count_images_with_class"][idx] += 1 if stat_count[class_name] > 0 else 0
            if class_name == "unlabeled":
                continue
            table_row.append(round(cur_area, 2))
            table_row.append(round(cur_count, 2))

        if len(table_row) != len(stats["columns"]):
            raise RuntimeError("Values for some columns are missed")

        stats["data"].append(table_row)

    @staticmethod
    def aggregate_calculations(stats: dict):
        # remove unnecessary dields
        stats.pop("sum_class_area_per_image", None)
        stats.pop("sum_class_count_per_image", None)
        stats.pop("count_images_with_class", None)
        stats.pop("class_names", None)
        stats.pop("class_indices_colors", None)
        stats.pop("_name_to_index", None)

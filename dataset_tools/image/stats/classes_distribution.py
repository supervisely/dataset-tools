import os
from dotenv import load_dotenv
import numpy as np

import supervisely as sly

# import dtoolz.image.stats.sly_globals as g


# import dtoolz as dtz


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()

PROJECT_ID = sly.env.project_id()
BG_COLOR = [0, 0, 0]

# stats = dtz.statistics.classes_distribution(project_id=777)
# stats = dtz.statistics.classes_distribution(project_path="/home/max/lemons")
# # to team files
# # or
# dtz.statistics.plot(stats)


def get_overviewTable(tb: dict, image_info, meta):
    # update_overview_table(image_info, meta)
    ann_info = api.annotation.download(
        image_info.id,
    )
    ann_json = ann_info.annotation
    ann_objects = [(obj["id"], obj["classTitle"]) for obj in ann_json["objects"]]

    ann = sly.Annotation.from_json(ann_json, meta)

    render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
    render_idx_rgb[:] = BG_COLOR

    ann.draw_class_idx_rgb(render_idx_rgb, tb["_name_to_index"])

    stat_area = sly.Annotation.stat_area(
        render_idx_rgb, tb["class_names"], tb["class_indices_colors"]
    )
    stat_count = ann.stat_class_count(tb["class_names"])

    if stat_area["unlabeled"] > 0:
        stat_count["unlabeled"] = 1

    for idx, class_name in enumerate(tb["class_names"]):
        cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
        cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0

        tb["sum_class_area_per_image"][idx] += cur_area
        tb["sum_class_count_per_image"][idx] += cur_count
        tb["count_images_with_class"][idx] += 1 if stat_count[class_name] > 0 else 0

        if class_name == "unlabeled":
            continue
        if stat_count[class_name] > 0:
            tb["image_counts_filter_by_id"][idx].append(image_info.id)
        if stat_count[class_name] > 0:
            obj_ids = [obj[0] for obj in ann_objects if obj[1] == class_name]
            tb["object_counts_filter_by_id"][idx].extend(obj_ids)

    ann = sly.Annotation.from_json(ann_json, meta)

    tb["images_count"] = tb["count_images_with_class"]
    tb["objects_count"] = tb["sum_class_count_per_image"]

    return tb


def update_classes_distribution(stats: dict, image_info):
    overviewTable = get_overviewTable(stats["overviewTable"], image_info, stats["meta"])
    stats["overviewTable"].update(overviewTable)
    # stats["b"] += 1


def classes_distribution(project_id: int = None, project_path: str = None):
    # explicit structure
    stats = {
        "overviewTable": {
            "class_names": [],
            "images_count": [],
            "image_counts_filter_by_id": [],
            "objects_count": [],
            "object_counts_filter_by_id": [],
            "avg_nonzero_area": [],
            "avg_nonzero_count": [],
        }
    }

    if project_id is not None:
        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)
        stats["meta"] = meta

        class_names = ["unlabeled"]
        class_colors = [[0, 0, 0]]
        class_indices_colors = [[0, 0, 0]]
        _name_to_index = {}
        for idx, obj_class in enumerate(meta.obj_classes):
            class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            class_index = idx + 1
            class_indices_colors.append([class_index, class_index, class_index])
            _name_to_index[obj_class.name] = class_index

        stats["overviewTable"]["class_names"] = class_names
        stats["overviewTable"]["class_indices_colors"] = class_indices_colors
        stats["overviewTable"]["_name_to_index"] = _name_to_index

        stats["overviewTable"]["sum_class_area_per_image"] = [0] * len(class_names)
        stats["overviewTable"]["sum_class_count_per_image"] = [0] * len(class_names)
        stats["overviewTable"]["count_images_with_class"] = [0] * len(class_names)

        stats["overviewTable"]["image_counts_filter_by_id"] = [[] for _ in class_names]
        stats["overviewTable"]["object_counts_filter_by_id"] = [[] for _ in class_names]

        for dataset in api.dataset.get_list(project_id):
            for image in api.image.get_list(dataset.id):
                update_classes_distribution(stats, image)

        # average nonzero class area per image
        with np.errstate(divide="ignore"):
            avg_nonzero_area = np.divide(
                stats["overviewTable"]["sum_class_area_per_image"],
                stats["overviewTable"]["count_images_with_class"],
            )
            avg_nonzero_count = np.divide(
                stats["overviewTable"]["sum_class_count_per_image"],
                stats["overviewTable"]["count_images_with_class"],
            )

        avg_nonzero_area = np.where(np.isnan(avg_nonzero_area), None, avg_nonzero_area)
        avg_nonzero_count = np.where(np.isnan(avg_nonzero_count), None, avg_nonzero_count)

        stats["overviewTable"]["avg_nonzero_area"] = avg_nonzero_area
        stats["overviewTable"]["avg_nonzero_count"] = avg_nonzero_count

    else:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)
        for dataset in project_fs:
            for image in dataset:
                update_classes_distribution(stats, image)
    return stats


stats = classes_distribution(project_id=PROJECT_ID)

print()

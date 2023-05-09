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


def get_overviewTable(table: dict, image_info, meta):
    # update_overview_table(image_info, meta)
    ann_info = api.annotation.download(
        image_info.id,
    )
    ann_json = ann_info.annotation

    ann = sly.Annotation.from_json(ann_json, meta)

    render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
    render_idx_rgb[:] = BG_COLOR

    stat_area = sly.Annotation.stat_area(
        render_idx_rgb, table["class_names"], table["class_indices_colors"]
    )
    stat_count = ann.stat_class_count(table["class_names"])

    if stat_area["unlabeled"] > 0:
        stat_count["unlabeled"] = 1

    for idx, class_name in enumerate(table["class_names"]):
        table["images_count"][idx] += 1 if stat_count[class_name] > 0 else 0

        cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0

        table["objects_count"][idx] += cur_count

    ann = sly.Annotation.from_json(ann_json, meta)

    return table


def update_classes_distribution(stats: dict, image_info):
    overviewTable = get_overviewTable(stats["overviewTable"], image_info, stats["meta"])
    stats["overviewTable"].update(overviewTable)
    # stats["b"] += 1


def classes_distribution(project_id: int = None, project_path: str = None):
    # structure
    stats = {
        "overviewTable": {
            "class_names": [],
            "images_count": [],
            "objects_count": [],
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
        for idx, obj_class in enumerate(meta.obj_classes):
            class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            class_index = idx + 1
            class_indices_colors.append([class_index, class_index, class_index])

        stats["overviewTable"]["class_names"] = class_names
        stats["overviewTable"]["class_indices_colors"] = class_indices_colors
        stats["overviewTable"]["images_count"] = [0] * len(class_names)
        stats["overviewTable"]["objects_count"] = [0] * len(class_names)

        for dataset in api.dataset.get_list(project_id):
            for image in api.image.get_list(dataset.id):
                update_classes_distribution(stats, image)
    else:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)
        for dataset in project_fs:
            for image in dataset:
                update_classes_distribution(stats, image)
    return stats


stats = classes_distribution(project_id=PROJECT_ID)

print()

import os
from dotenv import load_dotenv

import supervisely as sly
import dtoolz as dtz


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()

pass
"""
# stats = xxx.statistics.classes_distribution(project_id=777)
stats = dtz.statistics.classes_distribution(project_path="/home/max/lemons")
# to team files
# or
dtz.statistics.plot(stats)


def update_classes_distribution(stats: dict, image_info):
    stats["a"] += 1
    stats["b"] += 1


def classes_distribution(project_id, project_path):
    stats = {}
    if project_id is not None:
        for dataset in api.dataset.get_list():
            for image in api.image.get_list():
                update_classes_distribution(stats, image)
    else:
        project_fs = sly.Project(path, sly.OpenMode.READ)
        for dataset in project_fs:
            for image in dataset:
                update_classes_distribution(stats, image)
    return stats
"""

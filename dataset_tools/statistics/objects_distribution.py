import os
import json

from shutil import rmtree
from collections import namedtuple, defaultdict

import supervisely as sly

from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()

ImageData = namedtuple("ImageData", ["name", "ann"])

TEAM_ID = sly.io.env.team_id()
WORKSPACE_ID = sly.io.env.workspace_id()

SLY_APP_DATA_DIR = sly.app.get_data_dir()
os.makedirs(SLY_APP_DATA_DIR, exist_ok=True)

# Debug data.
PROJECT_ID = 21398
PROJECT_PATH = "/Users/iwatkot/Downloads/21281_dtools_test"

# Warning!
# Result JSON file will contain IMAGE NAMES IF STARTED FROM LOCAL PATH!
# If started from Supervisely project, result JSON file will contain IMAGE IDS.


def calculate_distribution(project_id: int = None, project_path: str = None):
    classes = defaultdict(lambda: defaultdict(lambda: {"count": 0, "images": []}))

    raw_data = []

    if project_path:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)

        project_meta = project_fs.meta
        project_id, project_name = project_fs.name.split("_", 1)

        print(
            f"Function launched for path: {project_path} with project_id: {project_id} with name {project_name}."
        )

        class_titles = [class_meta.name for class_meta in project_meta.obj_classes]

        datasets = project_fs.datasets

        print(f"Found {len(datasets)} datasets in project {project_name}.")

        for dataset in datasets:
            ann_dir = dataset.ann_dir
            image_names = [os.path.splitext(f)[0] for f in os.listdir(ann_dir)]

            ann_paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir)]
            anns_json = [json.load(open(ann_path)) for ann_path in ann_paths]

            raw_data.extend(list(zip(anns_json, image_names)))

        print(f"Generated raw_data dict with {len(raw_data)} items.")

    elif project_id:
        project_meta = get_project_meta(project_id)
        project_name = api.project.get_info_by_id(project_id).name

        print(
            f"Function launched for supervisely project_id: {project_id} with name {project_name}."
        )

        class_titles = [class_meta.name for class_meta in project_meta.obj_classes]

        datasets = api.dataset.get_list(project_id)

        print(f"Found {len(datasets)} datasets in project {project_name}.")

        for dataset in datasets:
            image_ids = [image_info.id for image_info in api.image.get_list(dataset.id)]
            ann_jsons = api.annotation.download_json_batch(dataset.id, image_ids)

            raw_data.extend(list(zip(ann_jsons, image_ids)))

        print(f"Generated raw_data dict with {len(raw_data)} items.")

    print("Starting iterating over raw_data dict.")

    for ann_json, image_id in raw_data:
        counters = defaultdict(lambda: {"count": 0, "image_ids": []})

        for obj in ann_json["objects"]:
            class_title = obj["classTitle"]
            counters[class_title]["count"] += 1
            counters[class_title]["image_ids"].append(image_id)

        for class_title in class_titles:
            count = counters[class_title]["count"]
            image_ids = counters[class_title]["image_ids"]
            classes[class_title][count]["images"].extend(list(set(image_ids)))
            classes[class_title][count]["count"] += 1

    print("Finished iterating over raw_data dict.")

    columns = set()
    rows = list()
    for class_title, class_data in classes.items():
        columns.update(class_data.keys())
        rows.append(class_title)

    print(f"Generated list of columns with {len(columns)} items.")
    print(f"Generated list of rows with {len(rows)} items.")

    distribution = {
        "project_id": project_id,
        "columns": list(columns),
        "rows": rows,
        "classes": classes,
    }

    save_path = os.path.join(SLY_APP_DATA_DIR, "distribution.json")

    print(f"Prepared distribution dict with ans will save it to {save_path}.")

    with open(save_path, "w") as f:
        json.dump(distribution, f, indent=4)

    return save_path


def get_project_meta(project_id):
    print(f"Trying to receive project meta from the API for project ID {project_id}.")

    project_meta_json = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta_json)

    print("Readed project meta and saved it in the global state.")

    return project_meta


def clean_tmp_dir():
    print("Cleaning tmp directory.")

    rmtree(SLY_APP_DATA_DIR, ignore_errors=True)
    os.makedirs(SLY_APP_DATA_DIR, exist_ok=True)


# Debug launchers.
# clean_tmp_dir()
# save_path = calculate_distribution(project_id=PROJECT_ID)
# save_path = calculate_distribution(project_path=PROJECT_PATH)
# print(f"Saved distribution to {save_path}")

import os
import json

from shutil import rmtree
from pprint import pprint
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
PROJECT_ID = 21281
PROJECT_PATH = "/Users/iwatkot/Downloads/21281_dtools_test"


def build_objects_table(project_id: int = None, project_path: str = None):
    raw_data = []

    if project_path:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)

        project_meta = project_fs.meta
        project_id, project_name = project_fs.name.split("_", 1)

        print(
            f"Function launched for path: {project_path} with project_id: {project_id} with name {project_name}."
        )

        datasets = project_fs.datasets

        print(f"Found {len(datasets)} datasets in project {project_name}.")

        for dataset in datasets:
            ann_dir = dataset.ann_dir
            image_names = [os.path.splitext(f)[0] for f in os.listdir(ann_dir)]

            ann_paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir)]
            anns_json = [json.load(open(ann_path)) for ann_path in ann_paths]

            anns = [
                sly.Annotation.from_json(ann_json, project_meta)
                for ann_json in anns_json
            ]

            dataset_data = {
                "dataset_name": dataset.name,
                "image_names": image_names,
                "anns": anns,
            }

            raw_data.append(dataset_data)

    elif project_id:
        project_meta = get_project_meta(project_id)
        project_name = api.project.get_info_by_id(project_id).name

        print(
            f"Function launched for supervisely project_id: {project_id} with name {project_name}."
        )

        datasets = api.dataset.get_list(project_id)

        for dataset in datasets:
            image_infos = api.image.get_list(dataset.id)

            image_names = [image_info.name for image_info in image_infos]
            image_ids = [image_info.id for image_info in image_infos]

            anns_json = api.annotation.download_json_batch(dataset.id, image_ids)
            anns = [
                sly.Annotation.from_json(ann_json, project_meta)
                for ann_json in anns_json
            ]

            dataset_data = {
                "dataset_name": dataset.name,
                "image_names": image_names,
                "anns": anns,
            }

            raw_data.append(dataset_data)

    print(f"Found {len(raw_data)} datasets in project {project_name}.")

    table_data = []

    for dataset_data in raw_data:
        for image_name, ann in zip(dataset_data["image_names"], dataset_data["anns"]):
            image_height, image_width = ann.img_size
            image_area = image_height * image_width

            for label in ann.labels:
                if type(label.geometry) not in [sly.Bitmap, sly.Rectangle, sly.Polygon]:
                    continue

                rect_geometry = label.geometry.to_bbox()

                height_px = rect_geometry.height
                height_pc = round(height_px * 100.0 / image_height, 2)

                width_px = rect_geometry.width
                width_pc = round(width_px * 100.0 / image_width, 2)

                area_px = label.geometry.area
                area_pc = round(area_px * 100.0 / image_area, 2)

                object_data = {
                    "object_id": label.geometry.sly_id,
                    "class_name": label.obj_class.name,
                    "image_name": image_name,
                    "dataset_name": dataset_data["dataset_name"],
                    "image_size_hw": f"{image_height}x{image_width}",
                    "height_px": height_px,
                    "height_pc": height_pc,
                    "width_px": width_px,
                    "width_pc": width_pc,
                    "area_px": area_px,
                    "area_pc": area_pc,
                }

                table_data.append(object_data)

    print(f"Prepared {len(table_data)} objects for the table.")

    save_path = os.path.join(SLY_APP_DATA_DIR, "objects_table.json")

    print(f"Prepared objects list and will save it to {save_path}.")

    with open(save_path, "w") as f:
        json.dump(table_data, f, indent=4)

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


clean_tmp_dir()
# save_path = build_objects_table(project_id=PROJECT_ID)
save_path = build_objects_table(project_path=PROJECT_PATH)
print(f"Saved objects table to {save_path}")

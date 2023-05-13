import json
import os
from collections import namedtuple
from shutil import rmtree

import numpy as np
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

MAX_WIDTH = 500
UPLOAD_DIR = "/renders"

# Debug data.
PROJECT_ID = 21281
PROJECT_PATH = "/Users/iwatkot/Downloads/21281_dtools_test"


def create_renders(project_id: int = None, project_path: str = None):
    raw_data = {}

    if project_path:
        project_fs = sly.Project(project_path, sly.OpenMode.READ)

        project_meta = project_fs.meta
        project_id, project_name = project_fs.name.split("_", 1)

        print(
            f"Function launched for path: {project_path} with project_id: {project_id} with name {project_name}."
        )

        datasets = project_fs.datasets

        for dataset in datasets:
            ann_dir = dataset.ann_dir
            image_names = [os.path.splitext(f)[0] for f in os.listdir(ann_dir)]

            ann_paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir)]
            anns_json = [json.load(open(ann_path)) for ann_path in ann_paths]

            raw_data[dataset.name] = list(zip(image_names, anns_json))

    elif project_id:
        project_name = api.project.get_info_by_id(project_id).name
        project_meta = get_project_meta(project_id)

        print(
            f"Function launched for supervisely project_id: {project_id} with name {project_name}"
        )

        datasets = api.dataset.get_list(project_id)

        print(f"Retrieved {len(datasets)} datasets. Starting iteration over datasets.")

        for dataset in datasets:
            image_ids = [image_info.id for image_info in api.image.get_list(dataset.id)]
            image_names = [image_info.name for image_info in api.image.get_list(dataset.id)]

            anns_json = api.annotation.download_json_batch(dataset.id, image_ids)

            raw_data[dataset.name] = list(zip(image_names, anns_json))

    for dataset_name, dataset_data in raw_data.items():
        print(f"Starting iteration over images in dataset: {dataset_name}.")

        for image_name, ann_json in dataset_data:
            image_ann = sly.Annotation.from_json(ann_json, project_meta)
            image_data = ImageData(image_name, image_ann)

            resize_and_save_image(project_id, project_name, dataset_name, image_data)

        print(f"Finished iteration over images in dataset: {dataset_name}.")

    print("Script finished creating renders.")

    upload_to_team_files()

    print("Script finished successfully.")


def upload_to_team_files():
    print(
        "Trying to upload renders to the team files. "
        f"Local directory: {SLY_APP_DATA_DIR}. Remote directory: {UPLOAD_DIR}."
    )

    api.file.upload_directory(TEAM_ID, SLY_APP_DATA_DIR, UPLOAD_DIR)

    print("Finished uploading renders to the team files.")


def resize_and_save_image(
    project_id: int, project_name: str, dataset_name: str, image_data: ImageData
):
    height, width = image_data.ann.img_size

    out_size = (int((height / width) * MAX_WIDTH), MAX_WIDTH)

    print(f"Calculated new annotation size (height, width): {out_size}.")

    resized_ann = image_data.ann.resize(out_size)

    print("Resized annotation.")

    image_filename = os.path.splitext(image_data.name)[0] + ".png"

    image_save_path = os.path.join(
        SLY_APP_DATA_DIR, f"{project_id}_{project_name}", dataset_name, image_filename
    )

    image = np.zeros((out_size[0], out_size[1], 3), dtype=np.uint8)

    resized_ann.draw_pretty(image, output_path=image_save_path, opacity=1)

    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0).tolist()

    print(f"Debug check of unique colors: {unique_colors}. Number of colors: {len(unique_colors)}.")

    print(f"Saved annotated image in {image_save_path}.")


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
# create_renders(project_id=PROJECT_ID)
# create_renders(project_path=PROJECT_PATH)

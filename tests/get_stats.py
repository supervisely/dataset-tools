import io
import json
import os
from tqdm import tqdm

import supervisely as sly
from dotenv import load_dotenv

import dataset_tools as dtools

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()

PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()
PROJECT_PATH = os.environ["LOCAL_DATA_DIR"]


import shutil

# if os.path.exists(PROJECT_PATH):
#     shutil.rmtree(PROJECT_PATH)

# sly.download(api, PROJECT_ID, PROJECT_PATH, save_image_info=True, save_images=False)


stats = {
    #    "spatial": dtools.stat.spation_distribution,
    "class_balance": dtools.ClassBalance,
    # "classesCoocccurence": dtools.ImgClassesCooccurence,
    # "classesPerImage": dtools.ClassesPerImage,
    #    "objects": dtools.stat.classes-on-every-image,
}

dtools.image.stats.calculate(
    api,
    stats,
    # project_id=PROJECT_ID,
    project_path=PROJECT_PATH,
    sample_rate=0.1,
    demo_dirpath="demo/",
)


print()

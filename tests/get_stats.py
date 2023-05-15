import io
import json
import os

import supervisely as sly
from dotenv import load_dotenv

import dataset_tools as dtools

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()


# import shutil

# shutil.rmtree(os.environ["LOCAL_DATA_DIR"])
# sly.download(api, PROJECT_ID, os.environ["LOCAL_DATA_DIR"], save_image_info=True)


stats = {
    #    "spatial": dtools.stat.spation_distribution,
    "class_balance": dtools.ClassBalance,
    # "classesCoocccurence": dtools.ImgClassesCooccurence,
    # "classesPerImage": dtools.ClassesPerImage,
    #    "objects": dtools.stat.classes-on-every-image,
}

dtools.image.stats.calculate(
    stats,
    project_id=PROJECT_ID,
    # project_dir=os.environ["LOCAL_DATA_DIR"],
    sample_rate=1,
    demo_dirpath="demo/",
)


print()

import io
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

import dataset_tools as dtools
import supervisely as sly

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


# stats = {
#     #    "spatial": dtools.stat.spation_distribution,
#     "class_balance": dtools.ClassBalance,
#     "classesCoocccurence": dtools.ImgClassesCooccurence,
#     # "classesPerImage": dtools.ClassesPerImage,
#     #    "objects": dtools.stat.classes-on-every-image,
# }


cls_balance = dtools.ClassBalance(project_meta)
stat_x = dtools.StaX(project_meta)
stat_y = dtools.StatY(project_meta)

dtools.get_stats(
    project_id=PROJECT_ID,
    # project_path=PROJECT_PATH,
    [cls_balance, stat_x, stat_y],
    sample_rate=0.1,
    # api (Optional) = None
)

stat_y.to_json() -> team_file - github -...


dtools.image.stats.calculate(
    api,
    stats,
    # project_id=PROJECT_ID,
    project_path=PROJECT_PATH,
    sample_rate=0.1,
    demo_dirpath="demo/",
)


print()

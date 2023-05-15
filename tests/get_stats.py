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


# project_meta, datasets = dtools.initialize(
#     project_id=PROJECT_ID,
#     # project_path=PROJECT_PATH,
# )


cls_balance = dtools.ClassBalance(project_meta)
cls_coocc = dtools.ClassCooccurence(project_meta)
# stat_y = dtools.StatY(project_meta)

dtools.get_stats(
    [
        # cls_ba/lance,
        cls_coocc,
    ],
    project_meta,
    datasets,
    sample_rate=1,
)

# cls_coocc.to_json()

for stat in [
    # cls_balance,
    cls_coocc,
]:
    demo_dirpath = "demo/"
    stat_name = type(stat).__name__
    os.makedirs(demo_dirpath, exist_ok=True)
    json_file_path = os.path.join(demo_dirpath, f"{stat_name}.json")
    image_file_path = os.path.join(demo_dirpath, f"{stat_name}.png")

    with open(json_file_path, "w") as f:
        json.dump(stat.to_json(), f)

    stat.to_image(image_file_path)

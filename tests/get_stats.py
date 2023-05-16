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


if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()
project_id = sly.env.project_id()
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

cls_perimage = dtools.ClassesPerImage(project_meta)
cls_balance = dtools.ClassBalance(project_meta)
cls_cooc = dtools.ClassCooccurence(project_meta)

obj_distrib = dtools.ObjectsDistribution(project_meta)
obj_sizes = dtools.ObjectSizes(project_meta)
cls_sizes = dtools.ClassSizes(project_meta)


dtools.count_stats(
    project_id,
    stats=[cls_perimage, cls_balance, cls_cooc, obj_distrib, obj_sizes, cls_sizes],
    sample_rate=0.1,
)

with open("./demo/class_perimage.json", "w") as f:
    json.dump(cls_perimage.to_json(), f)
# cls_perimage.to_image("./demo/class_perimage.png")
with open("./demo/class_balance.json", "w") as f:
    json.dump(cls_balance.to_json(), f)
cls_balance.to_image("./demo/class_balance.png")
with open("./demo/class_cooc.json", "w") as f:
    json.dump(cls_cooc.to_json(), f)
cls_cooc.to_image("./demo/class_cooc.png")

with open("./demo/obj_distrib.json", "w") as f:
    json.dump(obj_distrib.to_json(), f)
with open("./demo/object_sizes.json", "w") as f:
    json.dump(obj_sizes.to_json(), f)
with open("./demo/class_size.json", "w") as f:
    json.dump(cls_sizes.to_json(), f)

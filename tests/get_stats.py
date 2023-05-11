import os
from dotenv import load_dotenv

import supervisely as sly
import dataset_tools as dtools


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()


PROJECT_ID = sly.env.project_id()


import shutil

shutil.rmtree(os.environ["LOCAL_DATA_DIR"])
sly.download(api, PROJECT_ID, os.environ["LOCAL_DATA_DIR"], save_image_info=True)


cfg = {
    #    "spatial": dtools.stat.spation_distribution,
    "classesDistribution": dtools.ImgClassesDistribution,
    "classesCoocccurence": dtools.ImgClassesCooccurence,
    #     "images": dtools.stat.classes-on-every-image,
    #    "objects": dtools.stat.classes-on-every-image,
}

result = dtools.image.stats.calculate(
    api,
    cfg,
    # project_id=PROJECT_ID,
    project_dir=os.environ["LOCAL_DATA_DIR"],
    sample_rate=1,
)

print(result)

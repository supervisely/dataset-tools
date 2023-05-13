import io
import json
import os

import supervisely as sly
from dotenv import load_dotenv

import dataset_tools as dtools

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()


PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()
storage_dir = sly.app.get_data_dir()


import shutil

shutil.rmtree(os.environ["LOCAL_DATA_DIR"])
sly.download(api, PROJECT_ID, os.environ["LOCAL_DATA_DIR"], save_image_info=True)


cfg = {
    #    "spatial": dtools.stat.spation_distribution,
    "classesDistribution": dtools.ImgClassesDistribution,
    "classesCoocccurence": dtools.ImgClassesCooccurence,
    "classesPerImage": dtools.ClassesPerImage,
    #    "objects": dtools.stat.classes-on-every-image,
}

result = dtools.image.stats.calculate(
    api,
    cfg,
    project_id=PROJECT_ID,
    # project_dir=os.environ["LOCAL_DATA_DIR"],
    sample_rate=1,
)


for key, stats in result.items():
    # save stats to JSON file
    stat_json_path = os.path.join(storage_dir, f"{key}.json")
    with io.open(stat_json_path, "w", encoding="utf-8") as file:
        str_ = json.dumps(stats, indent=4, separators=(",", ": "), ensure_ascii=False)
        file.write(str(str_))

    # upload stats to Team files
    dst_path = f"/stats/{PROJECT_ID}/{key}.json"
    file_info = api.file.upload(TEAM_ID, stat_json_path, dst_path)

sly.fs.remove_dir(storage_dir)

print()

import os
import sys
from pathlib import Path

import supervisely as sly

# my_app = sly.AppService()
# api: sly.Api = my_app.public_api

root_source_dir = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

# TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()

logger = sly.logger

# train_ds = os.environ["modal.state.train"]
# test_ds = os.environ["modal.state.test"]

datasets = ["Train", "Test"]

# for ds in [train_ds, test_ds]:
#     if len(ds) != 2:
#         datasets.append(ds[1:-1].replace("'", ""))

# if len(datasets) == 0:
#     logger.warn("You have not selected a dataset to import")
#     my_app.stop()

train_percent = 100
test_percent = 100

sample_img_count = {"Train": round(6.7 * train_percent), "Test": round(3.31 * test_percent)}

project_name = "MinneApple"
work_dir = "apple_data"
apple_url = "https://conservancy.umn.edu/bitstream/handle/11299/206575/detection.tar.gz?sequence=2&isAllowed=y"

arch_name = "detection.tar.gz"
folder_name = "detection"
images_folder = "images"
anns_folder = "masks"
img_size = (1280, 720)
batch_size = 30
class_name = "apple"
train_ds = "Train"

obj_class = sly.ObjClass(class_name, sly.Bitmap)
obj_class_collection = sly.ObjClassCollection([obj_class])

meta = sly.ProjectMeta(obj_classes=obj_class_collection)

storage_dir = sly.app.get_data_dir()
work_dir_path = os.path.join(storage_dir, work_dir)
sly.io.fs.mkdir(work_dir_path)
archive_path = os.path.join(work_dir_path, arch_name)


import os, sys
from pathlib import Path
import supervisely as sly
from supervisely.app.v1.app_service import AppService


# my_app: AppService = AppService()
# api: sly.Api = my_app.public_api

root_source_dir = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

# TASK_ID = int(os.environ["TASK_ID"])
# TEAM_ID = int(os.environ['context.teamId'])
# WORKSPACE_ID = int(os.environ['context.workspaceId'])

logger = sly.logger

project_name = 'weed'
dataset_name = 'ds'
work_dir = 'weed'
weed_url = 'https://codeload.github.com/lameski/rgbweeddetection/zip/refs/heads/master'

arch_name = 'rgbweeddetection-master.zip'
folder_name = 'rgbweeddetection-master'

images_folder_name = 'Images'
annotation_folder_name = 'Weed_Plant_Masks'
ann_suffix = '_mask.png'
class_name_weed = 'weed'
class_name_carrot = 'carrot'

index_to_class = {'weed': 1, 'carrot': 2}
max_label_area = 3000

batch_size = 3

obj_class_weed = sly.ObjClass(class_name_weed, sly.Bitmap)
obj_class_carrot = sly.ObjClass(class_name_carrot, sly.Bitmap)
obj_class_collection = sly.ObjClassCollection([obj_class_weed, obj_class_carrot])

meta = sly.ProjectMeta(obj_classes=obj_class_collection)

storage_dir = sly.app.get_data_dir()
work_dir_path = os.path.join(storage_dir, work_dir)
sly.io.fs.mkdir(work_dir_path)
archive_path = os.path.join(work_dir_path, arch_name)
annotations_path = None
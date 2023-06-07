
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

project_name = 'cwfid'
dataset_name = 'ds'
work_dir = 'cwfid'
cwfid_url = 'https://codeload.github.com/cwfid/dataset/zip/refs/tags/v1.0'

arch_name = 'dataset-1.0.zip'
folder_name = 'dataset-1.0'
images_folder_name = 'images'
annotation_folder_name = 'annotations'

ann_suffix = '_annotation.yaml'
mask_suffix = '_annotation.png'

name_to_index = {'weed': 77, 'crop': 150}

class_name_plant = 'weed'
class_name_crop = 'crop'

batch_size = 30

obj_class_plant = sly.ObjClass(class_name_plant, sly.Bitmap)
obj_class_crop = sly.ObjClass(class_name_crop, sly.Bitmap)
obj_class_collection = sly.ObjClassCollection([obj_class_plant, obj_class_crop])

meta = sly.ProjectMeta(obj_classes=obj_class_collection)

storage_dir = sly.app.get_data_dir()
work_dir_path = os.path.join(storage_dir, work_dir)
sly.io.fs.mkdir(work_dir_path)
archive_path = os.path.join(work_dir_path, arch_name)
annotations_path = None
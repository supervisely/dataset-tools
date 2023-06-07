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
# TEAM_ID = int(os.environ['context.teamId'])
# WORKSPACE_ID = int(os.environ['context.workspaceId'])

logger = sly.logger

project_name = "Sweet Pepper"
dataset_name = "ds"
work_dir = "pepper"
strawberry_url = (
    "https://www.kaggle.com/datasets/lemontyc/sweet-pepper/download?datasetVersionNumber=1"
)

# (
#     "https://storage.googleapis.com/kaggle-data-sets/1746816/2853388/bundle/archive.zip?X-Goog-Algorithm="
#     "GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220315%"
#     "2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220315T120507Z&X-Goog-Expires=259199&X-Goog-"
#     "SignedHeaders=host&X-Goog-Signature=6731564b64e562feeaab4a51bd9845645e8743d51845856a37f07e5795642faa6"
#     "bdf73d1e423e9ba5873e58f31f404acf426200f29ada223fa8fc92f9d0b92c6f4e608c36e1d567348fcc8b3b232d1faae129d"
#     "1012e6c82f5896ce95f43f17def97ef56de3375d0914e7915ce80d0ff62b5450c4b5ac4e9114edfd9cf57dbe2c26f39315026"
#     "88ca04b4205f6bb77e6bba7b3cddbe46bba18f97798326345accb7e0e18a9cdf7fe48c54c017877acf3d0f1bab9cef324a890"
#     "c06c39ef6e52185c51bafb9e1292a5b5fab5ca5b99becad8965499228d016353019c338bf7e1fb115df9c5f19b9e3e9b846c4"
#     "ae85b4c5d4bf75f36b28261d257ca7ffe4ffc3b8965"
# )

arch_name = "archive.zip"
extract_folder_name = "dataset_620_red_yellow_cart_only"
annotation_file_name = "620_images_via_project.json"
images_ext = ".png"
sample_percent = round(int(100) * 6.2)
class_name = "pepper"

batch_size = 30

obj_class = sly.ObjClass(class_name, sly.Polygon)
obj_class_collection = sly.ObjClassCollection([obj_class])

color_tag_name = "color"
tag_meta_color = sly.TagMeta(color_tag_name, sly.TagValueType.ANY_STRING)
type_tag_name = "type"
tag_meta_type = sly.TagMeta(type_tag_name, sly.TagValueType.ANY_STRING)
tag_metas = [tag_meta_color, tag_meta_type]

tag_meta_collection = sly.TagMetaCollection(tag_metas)

meta = sly.ProjectMeta(obj_classes=obj_class_collection, tag_metas=tag_meta_collection)

storage_dir = sly.app.get_data_dir()
work_dir_path = os.path.join(storage_dir, work_dir)
sly.io.fs.mkdir(work_dir_path)
archive_path = os.path.join(work_dir_path, arch_name)
image_name_to_polygon = {}
image_name_to_attribute = {}

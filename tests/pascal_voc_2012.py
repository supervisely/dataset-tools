import os

import supervisely as sly
from dotenv import load_dotenv

import dataset_tools as dtools

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")


project_id = 223
teasm_id = 9
workspace_id = 28

api = sly.Api.from_env()
project_info = api.project.get_info_by_id(project_id)
print(project_info)

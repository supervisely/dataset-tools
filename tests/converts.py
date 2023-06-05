from dataset_tools.convert.cityscapes import to_supervisely, from_supervisely
import os
import supervisely as sly

from dotenv import load_dotenv

# if sly.is_development():
#     # load_dotenv(os.path.expanduser("~/ninja.env"))
#     load_dotenv(os.path.expanduser("~/supervisely.env"))
#     load_dotenv("local.env")

# api = sly.Api.from_env()

input_path = "/home/grokhi/downloads/reduced_cityscapes.tar"
# input_path = "/home/grokhi/downloads/reduced_cityscapes/"
# input_path = "/cityscapes/reduced_cityscapes.tar"

# res_path = to_supervisely(input_path)
# print(res_path)

res_path = "/home/grokhi/downloads/CITYSCAPES_TO_SLY"
res_path2 = from_supervisely(res_path, to_format="dir")
print(res_path2)

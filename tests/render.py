import os
import shutil

from dotenv import load_dotenv
from tqdm import tqdm

import dataset_tools as dtools
import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()
project_id = sly.env.project_id()
project_path = os.environ["LOCAL_DATA_DIR"]
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# if os.path.exists(project_path):
#     shutil.rmtree(project_path)

# sly.download(api, project_id, project_path, save_image_info=True)

poster = dtools.Poster(project_id, project_meta)
side_anns_grid = dtools.SideAnnotationsGrid(project_id, project_meta)
vertical_grid = dtools.VerticalGrid(project_id, project_meta)

dtools.prepare_renders(
    project_id,
    renderers=[poster, side_anns_grid, vertical_grid],
    sample_cnt=25,
)

poster_path = os.path.join(sly.app.get_data_dir(), "poster.png")
poster.to_image(poster_path)

side_anns_grid_path = os.path.join(sly.app.get_data_dir(), "grid_1.png")
side_anns_grid.to_image(side_anns_grid_path)

vertical_grid_path = os.path.join(sly.app.get_data_dir(), "v_grid.png")
vertical_grid.to_image(vertical_grid_path)

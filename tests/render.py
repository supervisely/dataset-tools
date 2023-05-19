import os

from dotenv import load_dotenv

import dataset_tools as dtools
import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

os.makedirs("./stats/", exist_ok=True)
api = sly.Api.from_env()

# 1. api way
project_id = sly.env.project_id()
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# 2. localdir way
project_path = os.environ["LOCAL_DATA_DIR"]
# sly.download(api, project_id, project_path, save_image_info=True, save_images=False)
# project_meta = sly.Project(project_path, sly.OpenMode.READ).meta

poster = dtools.Poster(project_path, project_meta)
side_anns_grid = dtools.SideAnnotationsGrid(project_path, project_meta)
horizontal_grid = dtools.HorizontalGrid(project_path, project_meta)

renderers = [
    poster,
    side_anns_grid,
    horizontal_grid,
]
dtools.prepare_renders(
    project_path,
    renderers=renderers,
    sample_cnt=40,
)

for r in renderers:
    r.to_image(f"./render_results/{r.basename_stem}.png")

import os

from dotenv import load_dotenv

import supervisely as sly
import dataset_tools as dtools

# only for tests
# if sly.is_development():
#     load_dotenv(os.path.expanduser("~/ninja.env"))
#     load_dotenv("local.env")

os.makedirs("./render_results/", exist_ok=True)
api = sly.Api.from_env()

# 1. api way
project_id = sly.env.project_id()
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# 2. localdir way
# project_path = os.environ["LOCAL_DATA_DIR"]
# sly.download(api, project_id, project_path, save_image_info=True, save_images=False)
# project_meta = sly.Project(project_path, sly.OpenMode.READ).meta


def main():
    renderers = [
        dtools.Poster(project_id, project_meta),
        dtools.SideAnnotationsGrid(project_id, project_meta),
    ]
    animators = [
        dtools.HorizontalGrid(project_id, project_meta),
        dtools.VerticalGrid(project_id, project_meta),
    ]

    # pass project_id or project_path as a first argument
    dtools.prepare_renders(
        project_id,
        renderers=renderers + animators,
        sample_cnt=40,
    )
    print("Saving visualizations...")
    for r in renderers + animators:
        r.to_image(f"./render_results/{r.basename_stem}.png")
    for a in animators:
        a.animate(f"./render_results/{a.basename_stem}.webm")
    print("Done.")


if __name__ == "__main__":
    main()

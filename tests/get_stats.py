import json
import os

from dotenv import load_dotenv

import supervisely as sly
import dataset_tools as dtools

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

os.makedirs("./stats/", exist_ok=True)
api = sly.Api.from_env()

# 1. api way
project_id = sly.env.project_id()
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
datasets = api.dataset.get_list(project_id)
project = api.project.get_info_by_id(project_id)

# 2. localdir way
# project_path = os.environ["LOCAL_DATA_DIR"]
# sly.download(api, project_id, project_path, save_image_info=True, save_images=False)
# project_meta = sly.Project(project_path, sly.OpenMode.READ).meta


def main():
    stats = [
        dtools.ClassesPerImage(project_meta, datasets),
        dtools.ClassBalance(project_meta),
        dtools.ClassCooccurrence(project_meta),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta),
        dtools.ClassSizes(project_meta),
    ]
    imstats = [
        dtools.ClassesHeatmaps(project_meta),
    ]
    vstats = [
        dtools.ClassesPreview(project_meta, project.name),
    ]

    # pass project_id or project_path as a first argument
    dtools.count_stats(
        project_id,
        stats=stats + imstats + vstats,
        sample_rate=0.01,
    )
    print("Saving stats...")
    for stat in stats:
        with open(f"./stats/{stat.basename_stem}.json", "w") as f:
            json.dump(stat.to_json(), f)
        stat.to_image(f"./stats/{stat.basename_stem}.png")
    for imstat in imstats:
        imstat.to_image(f"./stats/{imstat.basename_stem}.png", draw_style="outside_black")
    for vstat in vstats:
        vstat.animate(f"./render_results/originals/{vstat.basename_stem}.mp4")
    print("Converting files...")
    dtools.convert_all("render_results/originals")
    print("Done.")


if __name__ == "__main__":
    main()

import json
import os

from dotenv import load_dotenv

import supervisely as sly
import dataset_tools as dtools

# use creentials for debugging
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
project_stats = api.project.get_stats(project_id)

# 2. localdir way
# project_path = os.environ["LOCAL_DATA_DIR"]
# sly.download(api, project_id, project_path, save_image_info=True, save_images=False)
# project_meta = sly.Project(project_path, sly.OpenMode.READ).meta


def main():
    stats = [
        dtools.ClassesPerImage(project_meta, project_stats, datasets),
        dtools.ClassBalance(project_meta, project_stats),
        dtools.ClassCooccurrence(project_meta),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta, project_stats, datasets),
        dtools.ClassSizes(project_meta),
    ]

    # pass project_id or project_path as a first argument
    dtools.count_stats(
        project_id,
        stats=stats,
        sample_rate=0.01,
    )
    print("Saving stats...")
    for stat in stats:
        with open(f"./stats/{stat.basename_stem}.json", "w") as f:
            json.dump(stat.to_json(), f)
        stat.to_image(f"./stats/{stat.basename_stem}.png")


    heatmaps = dtools.ClassesHeatmaps(project_meta, project_stats)
    classes_previews = dtools.ClassesPreview(project_meta, project)
    vstats = [heatmaps, classes_previews]
    dtools.count_stats(
        project_id,
        stats=vstats,
        sample_rate=0.01,
    )
    heatmaps.to_image(f"./stats/{heatmaps.basename_stem}.png", draw_style="outside_black")
    classes_previews.animate(f"./stats/{classes_previews.basename_stem}.webm")
    
    print("Done.")


if __name__ == "__main__":
    main()

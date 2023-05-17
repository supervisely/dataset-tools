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

# 2. localdir way
# project_path = os.environ["LOCAL_DATA_DIR"]
# sly.download(api, project_id, project_path, save_image_info=True, save_images=False)
# project_meta = sly.Project(project_path, sly.OpenMode.READ).meta


def main():
    stats = [
        dtools.ClassesPerImage(project_meta),
        dtools.ClassBalance(project_meta),
        dtools.ClassCooccurrence(project_meta),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta),
        dtools.ClassSizes(project_meta),
        dtools.ClassesHeatmaps(project_meta),
    ]
    # pass project_id or project_path as a first argument
    dtools.count_stats(
        project_id,
        stats=stats,
        sample_rate=0.01,
    )
    print("Saving stats...")
    for stat in stats:
        if not isinstance(stat, dtools.ClassesHeatmaps):
            with open(f"./stats/{stat.json_name}.json", "w") as f:
                json.dump(stat.to_json(), f)
            stat.to_image(f"./stats/{stat.json_name}.png")
        else:
            stat.to_image("./stats/", ["inside_white", "outside_black"])


if __name__ == "__main__":
    main()

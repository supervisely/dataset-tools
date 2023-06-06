import os
import random
import zipfile
from typing import Literal

import gdown
from tqdm import tqdm

import dataset_tools.convert.laborotomato.sly_globals as g
import supervisely as sly
from supervisely.io.json import load_json_file


def prepare_ann_data(ann_path):
    ann_json = load_json_file(ann_path)
    annotations = ann_json["annotations"]
    images = ann_json["images"]

    for image_data in images:
        g.image_name_to_id[image_data["file_name"]] = image_data["id"]
        g.name_to_size[image_data["file_name"]] = (image_data["height"], image_data["width"])

    for ann_data in annotations:
        g.id_to_segm_anns[ann_data["image_id"]].append(ann_data["segmentation"][0])
        g.id_to_tag[ann_data["image_id"]].append(ann_data["category_id"])

    for category in ann_json["categories"]:
        g.category_id_to_name[category["id"]] = category["name"]


def create_ann(img_name):
    labels = []

    im_id = g.image_name_to_id[img_name]
    img_size = g.name_to_size[img_name]
    segm_anns = g.id_to_segm_anns[im_id]
    tag_ids = g.id_to_tag[im_id]

    for idx, segm_ann in enumerate(segm_anns):
        points = []
        for i in range(0, len(segm_ann), 2):
            points.append(sly.PointLocation(segm_ann[i + 1], segm_ann[i]))
        polygon = sly.Polygon(points, interior=[])

        tag_name = g.category_id_to_name[tag_ids[idx]]
        tag = sly.Tag(g.meta.get_tag_meta(tag_name))

        label = sly.Label(polygon, g.obj_class, tags=sly.TagCollection([tag]))
        labels.append(label)

    return sly.Annotation(img_size=img_size, labels=labels)


def extract_zip():
    if zipfile.is_zipfile(g.archive_path):
        with zipfile.ZipFile(g.archive_path, "r") as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn("Archive cannot be unpacked {}".format(g.arch_name))
        g.my_app.stop()


def to_supervisely(api):
    if not os.path.exists(g.archive_path):
        gdown.download(g.apple_url, g.archive_path, quiet=False)
    extract_zip()

    tomato_data_path = os.path.join(g.work_dir_path, g.folder_name)

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    for ds in g.datasets:
        new_dataset = api.dataset.create(new_project.id, ds, change_name_if_conflict=True)

        curr_img_path = os.path.join(tomato_data_path, ds.lower())
        curr_ann_path = os.path.join(tomato_data_path, g.anns_folder, ds.lower() + ".json")
        prepare_ann_data(curr_ann_path)

        curr_img_cnt = g.sample_img_count[ds]
        sample_img_path = random.sample(os.listdir(curr_img_path), curr_img_cnt)

        # progress = sly.Progress("Create dataset {}".format(ds), curr_img_cnt, app_logger)
        with tqdm(desc="Create dataset {}".format(ds), total=curr_img_cnt) as pbar:
            for img_batch in sly.batched(sample_img_path, batch_size=g.batch_size):
                img_pathes = [os.path.join(curr_img_path, name) for name in img_batch]
                img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(img_name) for img_name in img_batch]
                api.annotation.upload_anns(img_ids, anns)

                # progress.iters_done_report(len(img_batch))
                pbar.update(len(img_batch))

    return new_project.id


# def main():
#     sly.logger.info(
#         "Script arguments", extra={"TEAM_ID": g.TEAM_ID, "WORKSPACE_ID": g.WORKSPACE_ID}
#     )
#     g.my_app.run(initial_events=[{"command": "import_minne_apple"}])


# if __name__ == "__main__":
#     sly.main_wrapper("main", main)


def from_supervisely(
    input_path: str, output_path: str = None, to_format: Literal["dir", "tar", "both"] = "both"
) -> str:
    pass

import os
import random
import zipfile
from typing import Literal

import gdown
from tqdm import tqdm

import dataset_tools.convert.tomatod.sly_globals as g
import supervisely as sly
from supervisely.io.fs import get_file_name
from supervisely.io.json import load_json_file


def prepare_ann_data(ann_path):
    ann_json = load_json_file(ann_path)
    annotations = ann_json["annotations"]
    images = ann_json["images"]

    for image_data in images:
        g.image_name_to_id[image_data["file_name"]] = image_data["id"]
        g.name_to_size[image_data["file_name"]] = (image_data["height"], image_data["width"])

    for ann_data in annotations:
        g.id_to_bbox_anns[ann_data["image_id"]].append(ann_data["bbox"])
        g.id_to_tag[ann_data["image_id"]].append(ann_data["category_id"])

    for category in ann_json["categories"]:
        g.category_id_to_name[category["id"]] = category["name"]


def create_ann(img_name, ds_name):
    labels = []

    im_id = g.image_name_to_id[img_name]
    img_size = g.name_to_size[img_name]
    bbox_anns = g.id_to_bbox_anns[im_id]
    tag_ids = g.id_to_tag[im_id]

    for idx, bbox in enumerate(bbox_anns):
        rectangle = sly.Rectangle(bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2])

        if ds_name == g.train_ds:
            tag_name = g.category_id_to_name[tag_ids[idx]]
        else:  # test_ann diff from train_ann in tags data
            tag_name = g.category_id_to_name[1][tag_ids[idx] - 1]
        tag = sly.Tag(g.meta.get_tag_meta(tag_name))

        label = sly.Label(rectangle, g.obj_class, tags=sly.TagCollection([tag]))
        labels.append(label)

    return sly.Annotation(img_size=img_size, labels=labels)


def extract_zip(archive_path):
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn("Archive cannot be unpacked {}".format(get_file_name(archive_path)))
        g.my_app.stop()


def to_supervisely(api, WORKSPACE_ID):
    if not os.path.exists(g.images_archive_path):
        gdown.download(g.images_url, g.images_archive_path, quiet=False)
    extract_zip(g.images_archive_path)
    if not os.path.exists(g.annotations_archive_path):
        gdown.download(g.annotations_url, g.annotations_archive_path, quiet=False)
    extract_zip(g.annotations_archive_path)

    anns_path = os.path.join(g.work_dir_path, g.anns_folder)

    new_project = api.project.create(WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    for ds in g.datasets:
        new_dataset = api.dataset.create(new_project.id, ds, change_name_if_conflict=True)

        curr_img_path = os.path.join(g.work_dir_path, ds.lower())
        curr_ann_path = os.path.join(anns_path, g.ann_prefix + ds.lower() + g.ann_ext)
        prepare_ann_data(curr_ann_path)

        curr_img_cnt = g.sample_img_count[ds]
        sample_img_path = random.sample(os.listdir(curr_img_path), curr_img_cnt)

        with tqdm(desc="Create dataset {}".format(ds), total=curr_img_cnt) as pbar:
            for img_batch in sly.batched(sample_img_path, batch_size=g.batch_size):
                img_pathes = [os.path.join(curr_img_path, name) for name in img_batch]
                img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(img_name, ds) for img_name in img_batch]
                api.annotation.upload_anns(img_ids, anns)

                pbar.update(len(img_batch))

    return new_project.id


def from_supervisely(
    input_path: str, output_path: str = None, to_format: Literal["dir", "tar", "both"] = "both"
) -> str:
    pass

import zipfile, os, random, csv
import supervisely as sly
import sly_globals as g
from supervisely.io.fs import get_file_name, get_file_name_with_ext
import numpy as np
import gdown
from cv2 import connectedComponents
from typing import Literal


def download_and_extract():
    gdown.download(g.coffee_url, g.archive_path, quiet=False)
    if zipfile.is_zipfile(g.archive_path):
        with zipfile.ZipFile(g.archive_path, "r") as archive:
            archive.extractall(g.work_dir_path)
    else:
        sly.logger.warn("Archive cannot be unpacked {}".format(g.arch_name))


def read_csv(file_path):
    with open(file_path, newline="") as File:
        reader = csv.reader(File)
        for row in reader:
            g.tags_data[row[0]] = row[1:]


def get_tags(file_id):
    curr_tags = g.tags_data[file_id]
    tags = []
    for idx, curr_tag_val in enumerate(curr_tags):
        if int(curr_tag_val) == 0:
            continue

        tag = sly.Tag(g.tag_metas[idx], value=int(curr_tag_val))
        tags.append(tag)

    return sly.TagCollection(tags)


def create_annotation(ann_path):
    ann_np_leaf = sly.image.read(ann_path)[:, :, g.leaf_idx]
    mask_leaf = ann_np_leaf == g.obj_class_color_idxs[g.leaf_idx]
    bitmap = sly.Bitmap(mask_leaf)
    label = sly.Label(bitmap, g.obj_classes[g.leaf_idx])
    labels = [label]

    ann_np_symptom = sly.image.read(ann_path)[:, :, g.symptom_idx]
    mask_symptom = ann_np_symptom == g.obj_class_color_idxs[g.symptom_idx]
    if len(np.unique(mask_symptom)) != 1:
        ret, curr_mask = connectedComponents(mask_symptom.astype("uint8"), connectivity=8)
        for i in range(1, ret):
            obj_mask = curr_mask == i
            bitmap = sly.Bitmap(obj_mask)
            label = sly.Label(bitmap, g.obj_classes[g.symptom_idx])
            labels.append(label)

    file_id = get_file_name_with_ext(ann_path).split(g.ann_ext)[0]
    tags = get_tags(file_id)

    return sly.Annotation(
        img_size=(ann_np_leaf.shape[0], ann_np_leaf.shape[1]), labels=labels, img_tags=tags
    )


def to_supervisely(api: sly.Api):
    download_and_extract()

    coffee_data_path = os.path.join(g.work_dir_path, sly.fs.get_file_name(g.arch_name))
    tags_file = os.path.join(coffee_data_path, g.symptom_tag_file)
    read_csv(tags_file)

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    for ds in g.datasets:
        new_dataset = api.dataset.create(new_project.id, ds, change_name_if_conflict=True)

        curr_img_path = os.path.join(coffee_data_path, g.images_folder, ds.lower())
        curr_ann_path = os.path.join(coffee_data_path, g.anns_folder, ds.lower())

        curr_img_cnt = g.sample_img_count[ds]
        sample_img_path = random.sample(os.listdir(curr_img_path), curr_img_cnt)

        progress = sly.Progress("Create dataset {}".format(ds), curr_img_cnt, sly.logger)
        for img_batch in sly.batched(sample_img_path, batch_size=g.batch_size):
            img_pathes = [os.path.join(curr_img_path, name) for name in img_batch]
            ann_pathes = [
                os.path.join(curr_ann_path, get_file_name(name) + g.ann_ext) for name in img_batch
            ]

            anns = [create_annotation(ann_path) for ann_path in ann_pathes]

            img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
            img_ids = [im_info.id for im_info in img_infos]
            api.annotation.upload_anns(img_ids, anns)
            progress.iters_done_report(len(img_batch))

    return new_project.id


def from_supervisely(
    input_path: str, output_path: str = None, to_format: Literal["dir", "tar", "both"] = "both"
) -> str:
    pass

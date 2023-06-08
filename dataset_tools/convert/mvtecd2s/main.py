import os
import numpy as np
import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext, get_file_name, get_file_ext
from supervisely.io.json import load_json_file
from collections import defaultdict
import pycocotools.mask as mask_util
from typing import Literal


def to_supervisely(api: sly.Api, workspace_id):
    project_name = "MVTEC D2S"
    dataset_path = "MVTEC_D2S"
    batch_size = 30
    bbox_suffix = "_bbox"

    images_folder = "images"
    annotations_folder = "annotations"

    def convert_rle_mask_to_polygon(rle_mask_data):
        if type(rle_mask_data["counts"]) is str:
            rle_mask_data["counts"] = bytes(rle_mask_data["counts"], encoding="utf-8")
            mask = mask_util.decode(rle_mask_data)
        else:
            rle_obj = mask_util.frPyObjects(
                rle_mask_data,
                rle_mask_data["size"][0],
                rle_mask_data["size"][1],
            )
            mask = mask_util.decode(rle_obj)
        mask = np.array(mask, dtype=bool)
        return sly.Bitmap(mask).to_contours()

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        if annotations is not None:
            ann_data = image_name_to_ann[get_file_name_with_ext(image_path)]
            for curr_data in ann_data:
                if len(curr_data) != 0:
                    obj_class_poly, obj_class_rect = idx_to_obj_class[curr_data[0]]

                    rle_mask_data = curr_data[1]
                    polygons = convert_rle_mask_to_polygon(rle_mask_data)
                    for polygon in polygons:
                        label = sly.Label(polygon, obj_class_poly)
                        labels.append(label)

                    bbox_data = list(map(int, curr_data[2]))

                    left = bbox_data[0]
                    right = bbox_data[0] + bbox_data[2]
                    top = bbox_data[1]
                    bottom = bbox_data[1] + bbox_data[3]
                    rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                    label = sly.Label(rectangle, obj_class_rect)
                    labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    idx_to_obj_class = {}
    ds_exists_names = {}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta()

    annotations_path = os.path.join(dataset_path, annotations_folder)
    all_ann_data = os.listdir(annotations_path)

    for curr_ann_data in all_ann_data:
        curr_tag_meta = None
        tags = []
        file_name = get_file_name(curr_ann_data)
        ds_name = file_name.split("_")[1]
        file_name_prefix = file_name.split("D2S_" + ds_name + "_")
        if len(file_name_prefix) == 2:
            curr_tag_name = file_name_prefix[1]
            curr_tag_meta = sly.TagMeta(curr_tag_name, sly.TagValueType.NONE)
            tags = [sly.Tag(curr_tag_meta)]
            meta = meta.add_tag_meta(curr_tag_meta)

        if ds_name not in list(ds_exists_names.keys()):
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
            ds_exists_names[ds_name] = dataset.id

        curr_json_data = load_json_file(os.path.join(annotations_path, curr_ann_data))
        for data in curr_json_data["categories"]:
            if data["id"] not in list(idx_to_obj_class.keys()):
                obj_class = sly.ObjClass(data["name"], sly.Polygon)
                obj_class_rect = sly.ObjClass(data["name"] + bbox_suffix, sly.Rectangle)
                idx_to_obj_class[data["id"]] = (obj_class, obj_class_rect)
                meta = meta.add_obj_classes([obj_class, obj_class_rect])
        api.project.update_meta(project.id, meta.to_json())

        image_id_to_name = {}
        image_name_to_ann = defaultdict(list)
        for image_data in curr_json_data["images"]:
            image_id_to_name[image_data["id"]] = image_data["file_name"]

        annotations = curr_json_data.get("annotations")
        if annotations is not None:
            for ann_data in curr_json_data["annotations"]:
                image_name = image_id_to_name[ann_data["image_id"]]
                image_name_to_ann[image_name].append(
                    [ann_data["category_id"], ann_data["segmentation"], ann_data["bbox"]]
                )

        progress = sly.Progress("Add data to {} dataset".format(ds_name), len(image_id_to_name))

        for img_names_batch in sly.batched(list(image_id_to_name.values()), batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(dataset_path, images_folder, image_name)
                for image_name in img_names_batch
            ]

            checked_images_names_batch = []
            for img_name in img_names_batch:
                img_name = (
                    get_file_name(img_name)
                    + "_"
                    + get_file_name(curr_ann_data)
                    + get_file_ext(img_name)
                )
                checked_images_names_batch.append(img_name)

            img_infos = api.image.upload_paths(
                ds_exists_names[ds_name], checked_images_names_batch, images_pathes_batch
            )
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project.id


def from_supervisely(
    input_path: str, output_path: str = None, to_format: Literal["dir", "tar", "both"] = "both"
) -> str:
    pass

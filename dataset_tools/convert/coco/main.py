import json
import os
import shutil
from datetime import datetime
from typing import List

import numpy as np
import pycocotools.mask as mask_util
import requests
from dotenv import load_dotenv
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR


images_links = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "test2014": "http://images.cocodataset.org/zips/test2014.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
}

annotations_links = {
    "trainval2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

data_dir = sly.app.get_data_dir()

COCO_BASE_DIR = os.path.join(data_dir, "COCO")
META = sly.ProjectMeta()
img_dir = None
ann_dir = None
src_img_dir = None
dst_img_dir = None


def to_supervisely(
    original_ds_names: List[str] = None,
    custom_ds_paths: List[str] = None,
    dst_path: str = None,
):
    global img_dir, ann_dir, src_img_dir, dst_img_dir, META
    if original_ds_names is not None and custom_ds_paths is not None:
        raise ValueError("Both original and custom arguments are given, but only one is allowed.")
    is_original = True if original_ds_names is not None else False
    dst_path = os.path.join(data_dir, "supervisely project") if dst_path is None else dst_path

    if is_original:
        coco_datasets_dirs = download_original_coco_dataset(original_ds_names)
        coco_datasets_name = original_ds_names
    else:
        coco_datasets_dirs = custom_ds_paths
        coco_datasets_name = [os.path.basename(ds) for ds in custom_ds_paths]

    for dataset_name, coco_dataset_dir in zip(coco_datasets_name, coco_datasets_dirs):
        if not sly.fs.dir_exists(coco_dataset_dir):
            sly.logger.info(f"File {coco_dataset_dir} has been skipped.")
            continue
        coco_ann_dir = os.path.join(coco_dataset_dir, "annotations")
        if not sly.fs.dir_exists(os.path.join(coco_dataset_dir, "images")):
            sly.logger.warn(
                "Incorrect input data. Folder with images must be named 'images'. See 'README' for more information."
            )
            continue

        coco_ann_path = get_ann_path(coco_ann_dir, dataset_name, is_original)
        if bool(os.path.exists(coco_ann_path) and os.path.isfile(coco_ann_path)):
            coco = COCO(annotation_file=coco_ann_path)
            categories = coco.loadCats(ids=coco.getCatIds())
            coco_images = coco.imgs
            coco_anns = coco.imgToAnns

            sly_dataset_dir = create_sly_dataset_dir(dst_path, dataset_name=dataset_name)
            img_dir = os.path.join(sly_dataset_dir, "img")
            ann_dir = os.path.join(sly_dataset_dir, "ann")
            meta = get_sly_meta_from_coco(META, dst_path, categories, dataset_name)

            ds_progress = tqdm(desc=f"Converting dataset: {dataset_name}", total=len(coco_images))

            for img_id, img_info in coco_images.items():
                img_ann = coco_anns[img_id]
                img_size = (img_info["height"], img_info["width"])
                ann = create_sly_ann_from_coco_annotation(
                    meta=meta,
                    coco_categories=categories,
                    coco_ann=img_ann,
                    image_size=img_size,
                )
                move_trainvalds_to_sly_dataset(
                    dataset_dir=coco_dataset_dir, coco_image=img_info, ann=ann
                )
                ds_progress.update(1)
        else:
            sly_dataset_dir = create_sly_dataset_dir(dst_path, dataset_name=dataset_name)
            src_img_dir = os.path.join(coco_dataset_dir, "images")
            dst_img_dir = os.path.join(sly_dataset_dir, "img")
            ann_dir = os.path.join(sly_dataset_dir, "ann")
            move_testds_to_sly_dataset(dataset=dataset_name)

    if is_original:
        sly.fs.remove_dir(COCO_BASE_DIR)
    sly.logger.info(f"COCO dataset converted to Supervisely project: {dst_path}")
    return dst_path


def from_supervisely(
    src_path: str,
    dst_path: str = None,
    only_annotated_images: bool = True,
    only_annotations: bool = True,
):
    src_path = unpack_if_archive(src_path)
    parent_dir = os.path.dirname(os.path.normpath(src_path))
    coco_base_dir = os.path.join(parent_dir, "coco project") if dst_path is None else dst_path
    project = sly.Project(src_path, sly.OpenMode.READ)
    meta = project.meta
    datasets = project.datasets

    meta = prepare_meta(meta)
    categories_mapping = get_categories_map_from_meta(meta)
    sly.fs.mkdir(coco_base_dir)
    label_id = 0

    for dataset in datasets:
        sly.logger.info(f"processing {dataset.name}...")
        coco_dataset_dir = os.path.join(coco_base_dir, dataset.name)
        img_dir, ann_dir = create_coco_dataset(coco_dataset_dir)

        coco_ann = {}
        images = os.listdir(dataset.img_dir)

        ds_progress = tqdm(desc=f"Converting dataset: {dataset.name}", total=len(images))
        for batch in sly.batched(images):
            tmp_anns = [dataset.get_ann(name, meta) for name in batch]
            image_names = []
            anns = []
            if only_annotated_images is True:
                for ann, img_name in zip(tmp_anns, batch):
                    if len(ann.labels) > 0:
                        anns.append(ann)
                        image_names.append(img_name)
            else:
                image_names.extend(batch)
                anns.extend(tmp_anns)

            if only_annotations is False:
                src_paths = [dataset.get_img_path(name) for name in image_names]
                dst_paths = [os.path.join(img_dir, name) for name in image_names]
                for src_path, dst in zip(src_paths, dst_paths):
                    shutil.copyfile(src_path, dst)

            anns = [convert_annotation(ann, meta) for ann in anns]
            user_name = "Supervisely"
            coco_ann, label_id = create_coco_annotation(
                meta,
                categories_mapping,
                dataset,
                user_name,
                image_names,
                anns,
                label_id,
                coco_ann,
                ds_progress,
            )
        with open(os.path.join(ann_dir, "instances.json"), "w") as file:
            json.dump(coco_ann, file)

        sly.logger.info(f"dataset {dataset.name} processed!")

    return coco_base_dir


def download_original_coco_dataset(datasets):
    datasets_dirs = []
    for dataset in datasets:
        dataset_dir = os.path.join(COCO_BASE_DIR, dataset)
        sly.fs.mkdir(dataset_dir)
        archive_path = f"{dataset_dir}.zip"
        download_coco_images(dataset, archive_path, dataset_dir)
        if not dataset.startswith("test"):
            download_coco_annotations(dataset, archive_path, dataset_dir)
        datasets_dirs.append(dataset_dir)
    return datasets_dirs


def download_coco_images(dataset, archive_path, save_path):
    link = images_links[dataset]
    response = requests.head(link, allow_redirects=True)
    sizeb = int(response.headers.get("content-length", 0))
    p = tqdm(desc=f"Downloading COCO images", total=sizeb, unit="B", unit_scale=True)
    if not sly.fs.file_exists(archive_path):
        sly.fs.download(link, archive_path, progress=p.update)
    shutil.unpack_archive(archive_path, save_path, format="zip")
    os.rename(os.path.join(save_path, dataset), os.path.join(save_path, "images"))
    sly.fs.silent_remove(archive_path)


def download_coco_annotations(dataset, archive_path, save_path):
    link = None
    ann_dir = os.path.join(save_path, "annotations")
    if dataset in ["train2014", "val2014"]:
        if os.path.exists(ann_dir):
            return
        link = annotations_links["trainval2014"]
    elif dataset in ["train2017", "val2017"]:
        if os.path.exists(ann_dir):
            return
        link = annotations_links["trainval2017"]
    response = requests.head(link, allow_redirects=True)
    sizeb = int(response.headers.get("content-length", 0))
    p = tqdm(desc=f"Downloading COCO anns", total=sizeb, unit="B", unit_scale=True)
    sly.fs.download(link, archive_path, progress=p.update)
    shutil.unpack_archive(archive_path, save_path, format="zip")
    for file in os.listdir(ann_dir):
        if file != f"instances_{dataset}.json":
            sly.fs.silent_remove(os.path.join(ann_dir, file))
    sly.fs.silent_remove(archive_path)


def get_ann_path(ann_dir, dataset_name, is_original):
    if is_original:
        return os.path.join(ann_dir, f"instances_{dataset_name}.json")
    else:
        return os.path.join(ann_dir, "instances.json")


def create_sly_dataset_dir(dst_path, dataset_name):
    dataset_dir = os.path.join(dst_path, dataset_name)
    sly.fs.mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    sly.fs.mkdir(img_dir)
    ann_dir = os.path.join(dataset_dir, "ann")
    sly.fs.mkdir(ann_dir)
    return dataset_dir


def get_sly_meta_from_coco(meta, dst_path, coco_categories, dataset_name):
    path_to_meta = os.path.join(dst_path, "meta.json")
    if not os.path.exists(path_to_meta):
        meta = dump_meta(meta, coco_categories, path_to_meta)
    elif dataset_name not in ["train2014", "val2014", "train2017", "val2017"]:
        meta = dump_meta(meta, coco_categories, path_to_meta)
    return meta


def dump_meta(meta, coco_categories, path_to_meta):
    meta = create_sly_meta_from_coco_categories(meta, coco_categories)
    meta_json = meta.to_json()
    sly.json.dump_json_file(meta_json, path_to_meta)
    return meta


def create_sly_meta_from_coco_categories(meta, coco_categories):
    colors = []
    for category in coco_categories:
        if category["name"] in [obj_class.name for obj_class in meta.obj_classes]:
            continue
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)
        obj_class = sly.ObjClass(category["name"], sly.AnyGeometry, new_color)
        meta = meta.add_obj_class(obj_class)
    return meta


def create_sly_ann_from_coco_annotation(meta, coco_categories, coco_ann, image_size):
    labels = []
    for object in coco_ann:
        name_cat_id_map = coco_category_to_class_name(coco_categories)
        obj_class_name = name_cat_id_map[object["category_id"]]
        obj_class = meta.get_obj_class(obj_class_name)
        if type(object["segmentation"]) is dict:
            polygons = convert_rle_mask_to_polygon(object)
            for polygon in polygons:
                figure = polygon
                label = sly.Label(figure, obj_class)
                labels.append(label)
        elif type(object["segmentation"]) is list and object["segmentation"]:
            figure = convert_polygon_vertices(object)
            label = sly.Label(figure, obj_class)
            labels.append(label)

        bbox = object["bbox"]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        rectangle = sly.Label(
            sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax), obj_class
        )
        labels.append(rectangle)
    return sly.Annotation(image_size, labels)


def coco_category_to_class_name(coco_categories):
    return {category["id"]: category["name"] for category in coco_categories}


def convert_rle_mask_to_polygon(coco_ann):
    if type(coco_ann["segmentation"]["counts"]) is str:
        coco_ann["segmentation"]["counts"] = bytes(
            coco_ann["segmentation"]["counts"], encoding="utf-8"
        )
        mask = mask_util.decode(coco_ann["segmentation"])
    else:
        rle_obj = mask_util.frPyObjects(
            coco_ann["segmentation"],
            coco_ann["segmentation"]["size"][0],
            coco_ann["segmentation"]["size"][1],
        )
        mask = mask_util.decode(rle_obj)
    mask = np.array(mask, dtype=bool)
    return sly.Bitmap(mask).to_contours()


def convert_polygon_vertices(coco_ann):
    for polygons in coco_ann["segmentation"]:
        exterior = polygons
        exterior = [exterior[i * 2 : (i + 1) * 2] for i in range((len(exterior) + 2 - 1) // 2)]
        exterior = [sly.PointLocation(height, width) for width, height in exterior]
        return sly.Polygon(exterior, [])


def move_trainvalds_to_sly_dataset(dataset_dir, coco_image, ann):
    image_name = coco_image["file_name"]
    ann_json = ann.to_json()
    sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image_name}.json"))
    coco_img_path = os.path.join(dataset_dir, "images", image_name)
    sly_img_path = os.path.join(img_dir, image_name)
    if sly.fs.file_exists(os.path.join(coco_img_path)):
        shutil.copy(coco_img_path, sly_img_path)


def move_testds_to_sly_dataset(dataset):
    ds_progress = tqdm(f"Converting dataset: {dataset}", len(os.listdir(src_img_dir)))
    for image in os.listdir(src_img_dir):
        src_image_path = os.path.join(src_img_dir, image)
        dst_image_path = os.path.join(dst_img_dir, image)
        shutil.copy(src_image_path, dst_image_path)
        im = Image.open(dst_image_path)
        width, height = im.size
        img_size = (height, width)
        ann = sly.Annotation(img_size)
        ann_json = ann.to_json()
        sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image}.json"))
        ds_progress.update(1)


def prepare_meta(meta):
    new_classes = []
    for obj_cls in meta.obj_classes:
        obj_cls: sly.ObjClass
        new_classes.append(obj_cls.clone(geometry_type=GET_GEOMETRY_FROM_STR("polygon")))

    meta = meta.clone(obj_classes=sly.ObjClassCollection(new_classes))
    return meta


def get_categories_map_from_meta(meta):
    obj_classes = meta.obj_classes
    categories_mapping = {}
    for idx, obj_class in enumerate(obj_classes):
        categories_mapping[obj_class.name] = idx + 1
    return categories_mapping


def create_coco_dataset(coco_dataset_dir):
    sly.fs.mkdir(os.path.join(coco_dataset_dir))
    img_dir = os.path.join(coco_dataset_dir, "images")
    sly.fs.mkdir(img_dir)
    ann_dir = os.path.join(coco_dataset_dir, "annotations")
    sly.fs.mkdir(ann_dir)
    return img_dir, ann_dir


def convert_annotation(ann: sly.Annotation, dst_meta):
    new_labels = []
    for lbl in ann.labels:
        new_cls = dst_meta.obj_classes.get(lbl.obj_class.name)
        if lbl.obj_class.geometry_type == new_cls.geometry_type:
            new_labels.append(lbl)
        else:
            converted_labels = lbl.convert(new_cls)
            new_labels.extend(converted_labels)
    return ann.clone(labels=new_labels)


def create_coco_annotation(
    meta,
    categories_mapping,
    dataset,
    user_name,
    image_names,
    anns,
    label_id,
    coco_ann,
    progress,
):
    date_created = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if len(coco_ann) == 0:
        coco_ann = dict(
            info=dict(
                description=dataset.name,
                url="None",
                version=str(1.0),
                year=dataset.name[-4:],
                contributor=user_name,
                date_created=date_created,
            ),
            licenses=[dict(url="None", id=0, name="None")],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            # type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=get_categories_from_meta(meta),  # supercategory, id, name
        )

    for image_name, ann in zip(image_names, anns):
        ann: sly.Annotation
        height, width = ann.img_size
        coco_ann["images"].append(
            dict(
                license="None",
                file_name=image_name,
                url="None",  # image_info.full_storage_url,  # coco_url, flickr_url
                height=height,
                width=width,
                date_captured=date_created,
                id=image_name,
            )
        )

        for label in ann.labels:
            segmentation = label.geometry.to_json()["points"]["exterior"]
            segmentation = coco_segmentation(segmentation)

            bbox = label.geometry.to_bbox().to_json()["points"]["exterior"]
            bbox = coco_bbox(bbox)

            label_id += 1
            coco_ann["annotations"].append(
                dict(
                    segmentation=[
                        segmentation
                    ],  # a list of polygon vertices around the object, but can also be a run-length-encoded (RLE) bit mask
                    area=label.geometry.area,  # Area is measured in pixels (e.g. a 10px by 20px box would have an area of 200)
                    iscrowd=0,  # Is Crowd specifies whether the segmentation is for a single object or for a group/cluster of objects
                    image_id=image_name,  # The image id corresponds to a specific image in the dataset
                    bbox=bbox,  # he COCO bounding box format is [top left x position, top left y position, width, height]
                    category_id=categories_mapping[
                        label.obj_class.name
                    ],  # The category id corresponds to a single category specified in the categories section
                    id=label_id,  # Each annotation also has an id (unique to all other annotations in the dataset)
                )
            )
        progress.update(1)
    return coco_ann, label_id


def coco_segmentation(segmentation):  # works only with external vertices for now
    segmentation = [float(coord) for sublist in segmentation for coord in sublist]
    return segmentation


def coco_bbox(bbox):
    bbox = [float(coord) for sublist in bbox for coord in sublist]
    x, y, max_x, max_y = bbox
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    return bbox


def get_categories_from_meta(meta):
    obj_classes = meta.obj_classes
    categories = []
    for idx, obj_class in enumerate(obj_classes):
        categories.append(
            dict(
                supercategory=obj_class.name,
                id=idx + 1,  # supercategory id
                name=obj_class.name,
            )
        )
    return categories

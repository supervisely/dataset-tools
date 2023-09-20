import json
import os
import shutil
from datetime import datetime
from typing import List

import cv2

from copy import deepcopy
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
import requests
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


COCO_BASE_DIR = None
META = None
img_dir = None
ann_dir = None
src_img_dir = None
dst_img_dir = None


def to_supervisely(
    original_ds_names: List[str] = None,
    custom_ds_paths: List[str] = None,
    dst_path: str = None,
):
    global img_dir, ann_dir, src_img_dir, dst_img_dir, META, COCO_BASE_DIR
    META = sly.ProjectMeta()
    COCO_BASE_DIR = os.path.join(sly.app.get_data_dir(), "COCO")
    if original_ds_names is not None and custom_ds_paths is not None:
        raise ValueError("Both original and custom arguments are given, but only one is allowed.")
    is_original = True if original_ds_names is not None else False
    dst_path = os.path.join(sly.app.get_data_dir(), "supervisely project") if dst_path is None else dst_path

    if is_original:
        coco_datasets_dirs = download_original_coco_dataset(original_ds_names)
        coco_datasets_name = original_ds_names
    else:
        coco_datasets_dirs = custom_ds_paths
        coco_datasets_name = [os.path.basename(ds) for ds in custom_ds_paths]

    for dataset_name, coco_dataset_dir in zip(coco_datasets_name, coco_datasets_dirs):
        current_dataset_images_cnt = 0
        if not sly.fs.dir_exists(coco_dataset_dir):
            sly.logger.info(f"File {coco_dataset_dir} has been skipped.")
            continue
        coco_ann_dir = os.path.join(coco_dataset_dir, "annotations")
        if not sly.fs.dir_exists(os.path.join(coco_dataset_dir, "images")):
            sly.logger.warn(
                "Incorrect input data. Folder with images must be named 'images'. See 'README' for more information."
            )
            continue

        coco_instances_ann_path, coco_captions_ann_path = get_ann_path(
            ann_dir=coco_ann_dir, dataset_name=dataset_name, is_original=is_original
        )
        if coco_instances_ann_path is not None:
            try:
                coco_instances = COCO(annotation_file=coco_instances_ann_path)
            except Exception as e:
                raise Exception(
                    f"Incorrect instances annotation file: {coco_instances_ann_path}: {e}"
                )
            categories = coco_instances.loadCats(ids=coco_instances.getCatIds())
            coco_images = coco_instances.imgs
            coco_anns = coco_instances.imgToAnns

            coco_captions = None
            if coco_captions_ann_path is not None and sly.fs.file_exists(coco_captions_ann_path):
                try:
                    coco_captions = COCO(annotation_file=coco_captions_ann_path)
                    for img_id, ann in coco_instances.imgToAnns.items():
                        ann.extend(coco_captions.imgToAnns[img_id])
                except:
                    coco_captions = None

            sly_dataset_dir = create_sly_dataset_dir(dst_path, dataset_name=dataset_name)
            img_dir = os.path.join(sly_dataset_dir, "img")
            ann_dir = os.path.join(sly_dataset_dir, "ann")
            add_captions = coco_captions is not None
            META = get_sly_meta_from_coco(META, dst_path, categories, dataset_name, add_captions)

            ds_progress = tqdm(desc=f"Converting dataset: {dataset_name}", total=len(coco_images))

            for img_id, img_info in coco_images.items():
                image_name = img_info["file_name"]
                if "/" in image_name:
                    image_name = os.path.basename(image_name)
                if sly.fs.file_exists(os.path.join(coco_dataset_dir, "images", image_name)):
                    img_ann = coco_anns[img_id]
                    img_size = (img_info["height"], img_info["width"])
                    ann = create_sly_ann_from_coco_annotation(
                        meta=META,
                        coco_categories=categories,
                        coco_ann=img_ann,
                        image_size=img_size,
                    )
                    move_trainvalds_to_sly_dataset(
                        dataset_dir=coco_dataset_dir, coco_image=img_info, ann=ann
                    )
                    current_dataset_images_cnt += 1
                ds_progress.update(1)
        else:
            sly_dataset_dir = create_sly_dataset_dir(dst_path, dataset_name=dataset_name)
            META = get_sly_meta_from_coco(META, dst_path, [], dataset_name, False)
            src_img_dir = os.path.join(coco_dataset_dir, "images")
            dst_img_dir = os.path.join(sly_dataset_dir, "img")
            ann_dir = os.path.join(sly_dataset_dir, "ann")
            current_dataset_images_cnt = move_testds_to_sly_dataset(dataset=dataset_name, images_cnt=current_dataset_images_cnt)
        if current_dataset_images_cnt == 0:
            sly.logger.warn(f"Dataset {dataset_name} has no images.")
            remove_empty_sly_dataset_dir(sly_base_dir=dst_path, dataset_name=dataset_name)
        else:
            sly.logger.info(f"Dataset {dataset_name} has been successfully converted.")

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
    sly.fs.unpack_archive(archive_path, save_path)
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
    sly.fs.unpack_archive(archive_path, save_path)
    for file in os.listdir(ann_dir):
        if file != f"instances_{dataset}.json" and file != f"captions_{dataset}.json":
            sly.fs.silent_remove(os.path.join(ann_dir, file))
    sly.fs.silent_remove(archive_path)


def create_sly_dataset_dir(dst_path, dataset_name):
    dataset_dir = os.path.join(dst_path, dataset_name)
    sly.fs.mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    sly.fs.mkdir(img_dir)
    ann_dir = os.path.join(dataset_dir, "ann")
    sly.fs.mkdir(ann_dir)
    return dataset_dir


def get_sly_meta_from_coco(meta, dst_path, coco_categories, dataset_name, add_captions):
    path_to_meta = os.path.join(dst_path, "meta.json")
    if not os.path.exists(path_to_meta):
        meta = dump_meta(meta, coco_categories, path_to_meta, add_captions)
    elif dataset_name not in ["train2014", "val2014", "train2017", "val2017"]:
        meta = dump_meta(meta, coco_categories, path_to_meta, add_captions)
    return meta


def dump_meta(meta, coco_categories, path_to_meta, add_captions):
    meta = create_sly_meta_from_coco_categories(meta, coco_categories, add_captions)
    meta_json = meta.to_json()
    sly.json.dump_json_file(meta_json, path_to_meta)
    return meta


def create_sly_meta_from_coco_categories(meta, coco_categories, add_captions):
    colors = []
    for category in coco_categories:
        if category["name"] in [obj_class.name for obj_class in meta.obj_classes]:
            continue
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)
        obj_class = sly.ObjClass(category["name"], sly.AnyGeometry, new_color)
        meta = meta.add_obj_class(obj_class)
    if add_captions:
        if meta.get_tag_meta("caption") is None:
            meta = meta.add_tag_meta(sly.TagMeta("caption", sly.TagValueType.ANY_STRING))
    return meta


def create_sly_ann_from_coco_annotation(meta, coco_categories, coco_ann, image_size):
    labels = []
    imag_tags = []
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    for object in coco_ann:
        curr_labels = []
        

        segm = object.get("segmentation")
        if segm is not None and len(segm) > 0:
            obj_class_name = name_cat_id_map[object["category_id"]]
            obj_class = meta.get_obj_class(obj_class_name)
            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(object)
                for polygon in polygons:
                    figure = polygon
                    label = sly.Label(figure, obj_class)
                    labels.append(label)
            elif type(segm) is list and object["segmentation"]:
                figures = convert_polygon_vertices(object, image_size)
                curr_labels.extend([sly.Label(figure, obj_class) for figure in figures])
        labels.extend(curr_labels)

        bbox = object.get("bbox")
        if bbox is not None and len(bbox) == 4:
            obj_class_name = name_cat_id_map[object["category_id"]]
            obj_class = meta.get_obj_class(obj_class_name)
            if len(curr_labels) > 1:
                for label in curr_labels:
                    bbox = label.geometry.to_bbox()
                    labels.append(sly.Label(bbox, obj_class))
            else:
                x, y, w, h = bbox
                rectangle = sly.Label(sly.Rectangle(y, x, y + h, x + w), obj_class)
                labels.append(rectangle)

        caption = object.get("caption")
        if caption is not None:
            imag_tags.append(sly.Tag(meta.get_tag_meta("caption"), caption))

    return sly.Annotation(image_size, labels=labels, img_tags=imag_tags)


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


def convert_polygon_vertices(coco_ann, image_size):
    polygons = coco_ann["segmentation"]
    if all(type(coord) is float for coord in polygons):
        polygons = [polygons]

    exteriors = []
    for polygon in polygons:
        polygon = [polygon[i * 2 : (i + 1) * 2] for i in range((len(polygon) + 2 - 1) // 2)]
        exteriors.append([(width, height) for width, height in polygon])

    interiors = {idx: [] for idx in range(len(exteriors))}
    id2del = []
    for idx, exterior in enumerate(exteriors):
        temp_img = np.zeros(image_size + (3,), dtype=np.uint8)
        geom = sly.Polygon([sly.PointLocation(y, x) for x, y in exterior])
        geom.draw_contour(temp_img, color=[255, 255, 255])
        im = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        for idy, exterior2 in enumerate(exteriors):
            if idx == idy or idy in id2del:
                continue
            results = [cv2.pointPolygonTest(contours[0], (x, y), False) > 0 for x, y in exterior2]
            # if results of True, then all points are inside or on contour
            if all(results):
                interiors[idx].append(deepcopy(exteriors[idy]))
                id2del.append(idy)

    # remove contours from exteriors that are inside other contours
    for j in sorted(id2del, reverse=True):
        del exteriors[j]

    figures = []
    for exterior, interior in zip(exteriors, interiors.values()):
        exterior = [sly.PointLocation(y, x) for x, y in exterior]
        interior = [[sly.PointLocation(y, x) for x, y in points] for points in interior]
        figures.append(sly.Polygon(exterior, interior))

    return figures


def move_trainvalds_to_sly_dataset(dataset_dir, coco_image, ann):
    image_name = coco_image["file_name"]
    if "/" in image_name:
        image_name = os.path.basename(image_name)
    ann_json = ann.to_json()
    coco_img_path = os.path.join(dataset_dir, "images", image_name)
    sly_img_path = os.path.join(img_dir, image_name)
    if sly.fs.file_exists(os.path.join(coco_img_path)):
        sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image_name}.json"))
        shutil.copy(coco_img_path, sly_img_path)


def move_testds_to_sly_dataset(dataset, images_cnt):
    ds_progress = tqdm(desc=f"Converting dataset: {dataset}", total=len(os.listdir(src_img_dir)))
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
        images_cnt += 1
    return images_cnt


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


def get_ann_path(ann_dir, dataset_name, is_original):
    import glob
    instances_ann, captions_ann = None, None
    if is_original:
        instances_ann = os.path.join(ann_dir, f"instances_{dataset_name}.json")
        if not (os.path.exists(instances_ann) and os.path.isfile(instances_ann)):
            instances_ann = None
        captions_ann = os.path.join(ann_dir, f"captions_{dataset_name}.json")
        if not (os.path.exists(captions_ann) and os.path.isfile(captions_ann)):
            captions_ann = None
    else:
        ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
        if len(ann_files) == 1:
            instances_ann, captions_ann = ann_files[0], None
            sly.logger.warn(
                "Oonly one .json annotation file found. "
                "It will be used for instances. "
            )

        elif len(ann_files) > 1:
            instances_anns = [ann_file for ann_file in ann_files if "instance" in ann_file]
            captions_anns = [ann_file for ann_file in ann_files if "caption" in ann_file]
            if len(instances_anns) == 1:
                instances_ann = instances_anns[0]
            if len(captions_anns) == 1:
                captions_ann = captions_anns[0]
            if (
                instances_ann == captions_anns
                or len(captions_anns) == 0
                or len(instances_anns) == 0
            ):
                instances_ann = ann_files[0]
                captions_ann = None
                sly.logger.warn(
                    "Found more than one .json annotation file. "
                    "Import captions option is enabled, but more than one .json annotation file found. "
                    "It will be used for instances. "
                    "If you want to import captions, please, specify captions annotation file name."
                )
    sly.logger.info(f"instances_ann: {instances_ann}")
    sly.logger.info(f"captions_ann: {captions_ann}")
    return instances_ann, captions_ann


def remove_empty_sly_dataset_dir(sly_base_dir, dataset_name):
    dataset_dir = os.path.join(sly_base_dir, dataset_name)
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

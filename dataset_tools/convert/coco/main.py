import os, shutil, requests
from typing import List
import supervisely as sly
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from PIL import Image
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    load_dotenv("local.env")

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
    is_original: bool = True,
    original_ds: List[str] = [],
    custom_ds: str = None,
    dst_path: str = None,
):
    global img_dir, ann_dir, src_img_dir, dst_img_dir, META
    if is_original:
        coco_datasets = download_original_coco_dataset(original_ds)
    else:
        coco_datasets = list(os.listdir(custom_ds))

    if dst_path is None:
        dst_path = os.path.join(data_dir, "supervisely")
    for dataset in coco_datasets:
        coco_dataset_dir = os.path.join(COCO_BASE_DIR, dataset)
        if not sly.fs.dir_exists(coco_dataset_dir):
            sly.logger.info(f"File {coco_dataset_dir} has been skipped.")
            continue
        coco_ann_dir = os.path.join(coco_dataset_dir, "annotations")
        if not sly.fs.dir_exists(os.path.join(coco_dataset_dir, "images")):
            sly.logger.warn(
                "Incorrect input data. Folder with images must be named 'images'. See 'README' for more information."
            )
            continue

        if check_dataset_for_annotation(dataset, coco_ann_dir, is_original):
            coco_ann_path = get_ann_path(coco_ann_dir, dataset, is_original)

            coco = COCO(annotation_file=coco_ann_path)
            categories = coco.loadCats(ids=coco.getCatIds())
            coco_images = coco.imgs
            coco_anns = coco.imgToAnns

            sly_dataset_dir = create_sly_dataset_dir(dst_path, dataset_name=dataset)
            img_dir = os.path.join(sly_dataset_dir, "img")
            ann_dir = os.path.join(sly_dataset_dir, "ann")
            meta = get_sly_meta_from_coco(META, dst_path, categories, dataset)

            ds_progress = tqdm(desc=f"Converting dataset: {dataset}", total=len(coco_images))

            for img_id, img_info in coco_images.items():
                img_ann = coco_anns[img_id]
                img_size = (img_info["height"], img_info["width"])
                ann = create_sly_ann_from_coco_annotation(
                    meta=meta,
                    coco_categories=categories,
                    coco_ann=img_ann,
                    image_size=img_size,
                )
                move_trainvalds_to_sly_dataset(dataset=dataset, coco_image=img_info, ann=ann)
                ds_progress.update(1)
        else:
            sly_dataset_dir = create_sly_dataset_dir(dst_path, dataset_name=dataset)
            src_img_dir = os.path.join(COCO_BASE_DIR, dataset, "images")
            dst_img_dir = os.path.join(sly_dataset_dir, "img")
            ann_dir = os.path.join(sly_dataset_dir, "ann")
            move_testds_to_sly_dataset(dataset=dataset)

    sly.logger.info(f"COCO dataset converted to Supervisely project: {dst_path}")
    return dst_path


def download_original_coco_dataset(datasets):
    # for dataset in datasets:
    #     dataset_dir = os.path.join(COCO_BASE_DIR, dataset)
    #     sly.fs.mkdir(dataset_dir)
    #     archive_path = f"{dataset_dir}.zip"
    #     download_coco_images(dataset, archive_path, dataset_dir)
    #     if not dataset.startswith("test"):
    #         download_coco_annotations(dataset, archive_path, dataset_dir)
    return datasets


def download_coco_images(dataset, archive_path, save_path):
    link = images_links[dataset]
    response = requests.head(link, allow_redirects=True)
    sizeb = int(response.headers.get("content-length", 0))
    p = tqdm(desc=f"Downloading COCO {dataset} dataset", total=sizeb, unit="B", unit_scale=True)
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
    sly.fs.download(link, archive_path)
    shutil.unpack_archive(archive_path, save_path, format="zip")
    for file in os.listdir(ann_dir):
        if file != f"instances_{dataset}.json":
            sly.fs.silent_remove(os.path.join(ann_dir, file))
    sly.fs.silent_remove(archive_path)


def check_dataset_for_annotation(dataset_name, ann_dir, is_original):
    if is_original:
        ann_path = os.path.join(ann_dir, f"instances_{dataset_name}.json")
    else:
        ann_path = os.path.join(ann_dir, "instances.json")
    return bool(os.path.exists(ann_path) and os.path.isfile(ann_path))


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


def move_trainvalds_to_sly_dataset(dataset, coco_image, ann):
    image_name = coco_image["file_name"]
    ann_json = ann.to_json()
    sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image_name}.json"))
    coco_img_path = os.path.join(COCO_BASE_DIR, dataset, "images", image_name)
    sly_img_path = os.path.join(img_dir, image_name)
    if sly.fs.file_exists(os.path.join(coco_img_path)):
        shutil.move(coco_img_path, sly_img_path)


def move_testds_to_sly_dataset(dataset):
    ds_progress = sly.Progress(
        f"Converting dataset: {dataset}",
        len(os.listdir(src_img_dir)),
        min_report_percent=1,
    )
    for image in os.listdir(src_img_dir):
        src_image_path = os.path.join(src_img_dir, image)
        dst_image_path = os.path.join(dst_img_dir, image)
        shutil.move(src_image_path, dst_image_path)
        im = Image.open(dst_image_path)
        width, height = im.size
        img_size = (height, width)
        ann = sly.Annotation(img_size)
        ann_json = ann.to_json()
        sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image}.json"))
        ds_progress.iter_done_report()


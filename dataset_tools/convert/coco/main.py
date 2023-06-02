import json, os, shutil, requests
from typing import List
import supervisely as sly
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from PIL import Image
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    # load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()

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


def from_supervisely(
    local: bool = True,
    project_id: int = None,
    project_path: str = None,
    only_annotated_images: bool = True,
    only_annotations: bool = True,
):
    if local is True and project_path is not None:
        project = sly.Project(project_path, sly.OpenMode.READ)
        meta = project.meta
        datasets = project.datasets
    elif local is False and project_id is not None:
        api = sly.Api.from_env()
        project = api.project.get_info_by_id(project_id)
        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)
        datasets = api.dataset.get_list(project_id)
    else:
        raise ValueError("If project is local set project path else ID.")
    meta = prepare_meta(meta)
    categories_mapping = get_categories_map_from_meta(meta)

    coco_base_dir = os.path.join(data_dir, project.name)
    if project_path.endswith(coco_base_dir):
        coco_base_dir = f"{coco_base_dir}-exported"
    sly.fs.mkdir(coco_base_dir)
    label_id = 0

    for dataset in datasets:
        sly.logger.info(f"processing {dataset.name}...")
        coco_dataset_dir = os.path.join(coco_base_dir, dataset.name)
        img_dir, ann_dir = create_coco_dataset(coco_dataset_dir)

        coco_ann = {}
        if local:
            dataset: sly.Dataset
            images = [
                dataset.get_image_info(sly.fs.get_file_name_with_ext(img))
                for img in os.listdir(dataset.img_dir)
            ]
            images = images[:1]
        else:
            images = api.image.get_list(dataset.id)

        if only_annotated_images is True:
            images = [image for image in images if image.labels_count > 0 or len(image.tags) > 0]

        ds_progress = tqdm(desc=f"Converting dataset: {dataset.name}", total=len(images))
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]

            if only_annotations is False:
                if local:
                    # src_paths = [dataset.get_img_path(img) for img in os.listdir(dataset.img_dir)]
                    # dst_paths = [os.path.join(img_dir, img) for img in os.listdir(dataset.img_dir)]
                    # for src_path, dst_path in zip(src_paths, dst_paths):
                    #     shutil.copyfile(src_path, dst_path)
                    shutil.copytree(dataset.img_dir, img_dir)
                else:
                    image_paths = [
                        os.path.join(coco_dataset_dir, img_dir, image_info.name)
                        for image_info in batch
                    ]
                    api.image.download_paths(dataset.id, image_ids, image_paths)

            if local:
                anns = [dataset.get_ann(img, meta) for img in os.listdir(dataset.img_dir)]
            else:
                anns_json = api.annotation.download_json_batch(dataset.id, image_ids)
                anns = [sly.Annotation.from_json(ann_json, meta) for ann_json in anns_json]
            anns = [convert_annotation(ann, meta) for ann in anns]
            user_name = "Supervisely"
            coco_ann, label_id = create_coco_annotation(
                local,
                meta,
                categories_mapping,
                dataset,
                user_name,
                batch,
                anns,
                label_id,
                coco_ann,
                ds_progress,
            )
        with open(os.path.join(ann_dir, "instances.json"), "w") as file:
            json.dump(coco_ann, file)

        sly.logger.info(f"dataset {dataset.name} processed!")


def download_original_coco_dataset(datasets):
    for dataset in datasets:
        dataset_dir = os.path.join(COCO_BASE_DIR, dataset)
        sly.fs.mkdir(dataset_dir)
        archive_path = f"{dataset_dir}.zip"
        download_coco_images(dataset, archive_path, dataset_dir)
        if not dataset.startswith("test"):
            download_coco_annotations(dataset, archive_path, dataset_dir)
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
    # sly.fs.mkdir(img_dir)
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
    local,
    meta,
    categories_mapping,
    dataset,
    user_name,
    image_infos,
    anns,
    label_id,
    coco_ann,
    progress,
):
    if len(coco_ann) == 0:
        coco_ann = dict(
            info=dict(
                description=dataset.name if local else dataset.description,
                url="None",
                version=str(1.0),
                year=dataset.name[-4:] if local else int(dataset.created_at[:4]),
                contributor=user_name,
                date_created=dataset.name[-4:] if local else dataset.created_at,
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

    for image_info, ann in zip(image_infos, anns):
        coco_ann["images"].append(
            dict(
                license="None",
                file_name=image_info.name,
                url="None",  # image_info.full_storage_url,  # coco_url, flickr_url
                height=image_info.height,
                width=image_info.width,
                date_captured=image_info.created_at,
                id=image_info.id,
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
                    image_id=image_info.id,  # The image id corresponds to a specific image in the dataset
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


# to_supervisely(original_ds=["val2017"])
project_path = "/private/tmp/sly_data_dir/COCO test2"
# project = api.project.get_info_by_id(20649)
# pr = tqdm(total=project.items_count)
# sly.download(
#     api, 20649, project_path, save_image_info=True, save_images=True, progress_cb=pr.update
# )
from_supervisely(project_path=project_path, only_annotations=False)

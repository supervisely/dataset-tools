import os, random
import zipfile, json, tarfile
import supervisely as sly
import glob
from supervisely.io.fs import download, file_exists, get_file_name, get_file_name_with_ext
from supervisely.imaging.color import generate_rgb
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

from typing import Literal
import shutil


from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.polygon import Polygon
from supervisely.io.json import dump_json_file
from supervisely.io.fs import mkdir, get_file_name, get_file_ext, silent_remove
from supervisely.imaging.image import write

from dataset_tools.convert import unpack_if_archive

# my_app = sly.AppService()
# TEAM_ID = int(os.environ["context.teamId"])
# WORKSPACE_ID = int(os.environ["context.workspaceId"])
# INPUT_DIR = os.environ.get("modal.state.slyFolder")
# INPUT_FILE = os.environ.get("modal.state.slyFile")
PROJECT_NAME = "cityscapes"
IMAGE_EXT = ".png"
logger = sly.logger
samplePercent = int(80) / 100
train_tag = "train"
trainval_tag = "trainval"
val_tag = "val"

city_classes_to_colors = {
    "unlabeled": (0, 0, 0),
    "ego vehicle": (98, 15, 138),
    "rectification border": (15, 120, 55),
    "out of roi": (125, 138, 15),
    "static": (63, 15, 138),
    "dynamic": (111, 74, 0),
    "ground": (81, 0, 81),
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "parking": (250, 170, 160),
    "rail track": (230, 150, 140),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "guard rail": (180, 165, 180),
    "bridge": (150, 100, 100),
    "tunnel": (150, 120, 90),
    "pole": (153, 153, 153),
    "polegroup": (153, 153, 153),
    "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "caravan": (0, 0, 90),
    "trailer": (0, 0, 110),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
    "license plate": (0, 0, 142),
}

city_colors = list(city_classes_to_colors.values())


def create_sly_dataset_dir(dst_path, dataset_name):
    dataset_dir = os.path.join(dst_path, dataset_name)
    sly.fs.mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    sly.fs.mkdir(img_dir)
    ann_dir = os.path.join(dataset_dir, "ann")
    sly.fs.mkdir(ann_dir)
    return dataset_dir


def json_path_to_image_path(json_path):
    img_path = json_path.replace("/gtFine/", "/leftImg8bit/")
    img_path = img_path.replace("_gtFine_polygons.json", "_leftImg8bit" + IMAGE_EXT)
    return img_path


def convert_points(simple_points):
    return [sly.PointLocation(int(p[1]), int(p[0])) for p in simple_points]


def get_split_idxs(num_imgs, percentage):
    train_sample_idxs = int(np.floor(num_imgs * percentage))
    random_idxs = random.sample(population=range(num_imgs), k=train_sample_idxs)
    return random_idxs


def to_supervisely(input_path: str, output_path: str = None):
    input_dir = unpack_if_archive(input_path)

    tag_metas = sly.TagMetaCollection()
    obj_classes = sly.ObjClassCollection()
    dataset_names = []

    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path), "CITYSCAPES_TO_SLY")

    out_project = sly.Project(output_path, sly.OpenMode.CREATE)
    tags_template = os.path.join(input_dir, "gtFine", "*")
    tags_paths = glob.glob(tags_template)
    tags = [os.path.basename(tag_path) for tag_path in tags_paths]
    if train_tag in tags and val_tag not in tags:
        split_train = True
    elif trainval_tag in tags and val_tag not in tags:
        split_train = True
    else:
        split_train = False

    search_fine = os.path.join(input_dir, "gtFine", "*", "*", "*_gt*_polygons.json")
    files_fine = glob.glob(search_fine)
    files_fine.sort()
    search_imgs = os.path.join(input_dir, "leftImg8bit", "*", "*", "*_leftImg8bit" + IMAGE_EXT)
    files_imgs = glob.glob(search_imgs)
    files_imgs.sort()
    if len(files_fine) == 0 or len(files_imgs) == 0:
        raise Exception("Input cityscapes format not correct")

    samples_count = len(files_fine)
    images_pathes_for_compare = []
    images_pathes = {}
    images_names = {}
    anns_data = {}

    if samples_count > 2:
        random_train_indexes = get_split_idxs(samples_count, samplePercent)
    with tqdm(desc="Converting images to sly format", total=samples_count) as pbar:
        for idx, orig_ann_path in enumerate(files_fine):
            parent_dir, json_filename = os.path.split(os.path.abspath(orig_ann_path))
            dataset_name = os.path.basename(parent_dir)
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)

                ds = out_project.create_dataset(dataset_name)

                images_pathes[dataset_name] = []
                images_names[dataset_name] = []
                anns_data[dataset_name] = []
            orig_img_path = json_path_to_image_path(orig_ann_path)
            images_pathes_for_compare.append(orig_img_path)
            if not file_exists(orig_img_path):
                logger.warn(
                    "Image for annotation {} not found is dataset {}".format(
                        orig_ann_path.split("/")[-1], dataset_name
                    )
                )
                continue
            images_pathes[dataset_name].append(orig_img_path)
            images_names[dataset_name].append(sly.io.fs.get_file_name_with_ext(orig_img_path))
            tag_path = os.path.split(parent_dir)[0]
            train_val_tag = os.path.basename(tag_path)
            if split_train is True and samples_count > 2:
                if (train_val_tag == train_tag) or (train_val_tag == trainval_tag):
                    if idx in random_train_indexes:
                        train_val_tag = train_tag
                    else:
                        train_val_tag = val_tag

            tag_meta = sly.TagMeta("split", sly.TagValueType.ANY_STRING)
            if not tag_metas.has_key(tag_meta.name):
                tag_metas = tag_metas.add(tag_meta)

            tag = sly.Tag(meta=tag_meta, value=train_val_tag)
            json_data = json.load(open(orig_ann_path))
            ann = sly.Annotation.from_img_path(orig_img_path)

            for obj in json_data["objects"]:
                class_name = obj["label"]
                if class_name == "out of roi":
                    polygon = obj["polygon"][:5]
                    interiors = [obj["polygon"][5:]]
                else:
                    polygon = obj["polygon"]
                    if len(polygon) < 3:
                        logger.warn(
                            "Polygon must contain at least 3 points in ann {}, obj_class {}".format(
                                orig_ann_path, class_name
                            )
                        )
                        continue
                    interiors = []
                interiors = [convert_points(interior) for interior in interiors]
                polygon = sly.Polygon(convert_points(polygon), interiors)
                if city_classes_to_colors.get(class_name, None):
                    obj_class = sly.ObjClass(
                        name=class_name,
                        geometry_type=sly.Polygon,
                        color=city_classes_to_colors[class_name],
                    )
                else:
                    new_color = generate_rgb(city_colors)
                    city_colors.append(new_color)
                    obj_class = sly.ObjClass(
                        name=class_name, geometry_type=sly.Polygon, color=new_color
                    )
                ann = ann.add_label(sly.Label(polygon, obj_class))
                if not obj_classes.has_key(class_name):
                    obj_classes = obj_classes.add(obj_class)
            ann = ann.add_tag(tag)
            anns_data[dataset_name].append(ann)
            ds.add_item_file(sly.fs.get_file_name_with_ext(orig_img_path), orig_img_path, ann=ann)

            pbar.update(1)

    out_meta = sly.ProjectMeta(
        obj_classes=obj_classes, tag_metas=tag_metas, project_type=sly.ProjectType.IMAGES.value
    )
    out_project.set_meta(out_meta)

    stat_dct = {
        "samples": samples_count,
        "src_ann_cnt": len(files_fine),
        "src_img_cnt": len(files_imgs),
    }
    logger.info("Found img/ann pairs.", extra=stat_dct)
    images_without_anns = set(files_imgs) - set(images_pathes_for_compare)
    if len(images_without_anns) > 0:
        logger.warn("Found source images without corresponding annotations:")
        for im_path in images_without_anns:
            logger.warn("Annotation not found {}".format(im_path))

    logger.info(
        "Found classes.",
        extra={
            "cnt": len(obj_classes),
            "classes": sorted([obj_class.name for obj_class in obj_classes]),
        },
    )
    logger.info(
        "Created tags.",
        extra={
            "cnt": len(out_meta.tag_metas),
            "tags": sorted([tag_meta.name for tag_meta in out_meta.tag_metas]),
        },
    )

    return output_path


# RESULT_DIR_NAME = "cityscapes_format"
images_dir_name = "leftImg8bit"
annotations_dir_name = "gtFine"
default_dir_train = "train"
default_dir_val = "val"
default_dir_test = "test"
cityscapes_images_suffix = "_leftImg8bit"
cityscapes_polygons_suffix = "_gtFine_polygons.json"
cityscapes_color_suffix = "_gtFine_color.png"
cityscapes_labels_suffix = "_gtFine_labelIds.png"
possible_geometries = [Bitmap, Polygon]
possible_tags = ["train", "val", "test"]
splitter_coef = 3 / 5
if splitter_coef > 1 or splitter_coef < 0:
    raise ValueError(
        "train_to_val_test_coef should be between 0 and 1, your data is {}".format(splitter_coef)
    )


def from_supervisely(
    input_path: str, output_path: str = None, to_format: Literal["dir", "tar", "both"] = "both"
) -> str:
    input_dir = unpack_if_archive(input_path)

    project_fs = sly.Project(input_dir, sly.OpenMode.READ)
    meta = project_fs.meta
    datasets = project_fs.datasets

    # api.file.list

    # api = sly.Api()

    storage_dir = os.path.dirname(input_dir)

    if not output_path:
        RESULT_DIR_NAME = "SLY_TO_CITYSCAPES"
        # output_path = os.path.join(os.path.dirname(input_path), "SLY_TO_CITYSCAPES")

    def get_image_and_ann():
        mkdir(image_dir_path)
        mkdir(ann_dir)
        image_path = os.path.join(image_dir_path, image_name)
        # api.image.download_path(image_id, image_path)
        image_ext_to_png(image_path)

        mask_color, mask_label, poly_json = from_ann_to_cityscapes_mask(
            ann, name2id, sly.logger, train_val_flag
        )
        # dump_json_file(poly_json,
        #                os.path.join(ann_dir, get_file_name(base_image_name) + cityscapes_polygons_suffix))
        # write(
        #     os.path.join(ann_dir,
        #                  get_file_name(base_image_name) + cityscapes_color_suffix), mask_color)
        # write(
        #     os.path.join(ann_dir,
        #                  get_file_name(base_image_name) + cityscapes_labels_suffix), mask_label)

        dump_json_file(
            poly_json,
            os.path.join(
                ann_dir,
                get_file_name(base_image_name).replace("_leftImg8bit", "")
                + cityscapes_polygons_suffix,
            ),
        )
        write(
            os.path.join(
                ann_dir,
                get_file_name(base_image_name).replace("_leftImg8bit", "")
                + cityscapes_color_suffix,
            ),
            mask_color,
        )
        write(
            os.path.join(
                ann_dir,
                get_file_name(base_image_name).replace("_leftImg8bit", "")
                + cityscapes_labels_suffix,
            ),
            mask_label,
        )

    ARCHIVE_NAME = "{}_cityscapes.tar".format(project_fs.name)

    has_bitmap_poly_shapes = False
    for obj_class in meta.obj_classes:
        if obj_class.geometry_type not in possible_geometries:
            sly.logger.warn(
                f"Cityscapes format supports only bitmap and polygon classes, {obj_class.geometry_type} will be skipped"
            )
        else:
            has_bitmap_poly_shapes = True

    if has_bitmap_poly_shapes is False:
        raise Exception("Input project does not contain bitmap or polygon classes")

    RESULT_ARCHIVE = os.path.join(storage_dir, ARCHIVE_NAME)
    RESULT_DIR = os.path.join(storage_dir, RESULT_DIR_NAME)
    result_images_train = os.path.join(RESULT_DIR, images_dir_name, default_dir_train)
    result_images_val = os.path.join(RESULT_DIR, images_dir_name, default_dir_val)
    result_images_test = os.path.join(RESULT_DIR, images_dir_name, default_dir_test)
    result_anns_train = os.path.join(RESULT_DIR, annotations_dir_name, default_dir_train)
    result_anns_val = os.path.join(RESULT_DIR, annotations_dir_name, default_dir_val)
    result_anns_test = os.path.join(RESULT_DIR, annotations_dir_name, default_dir_test)
    sly.fs.mkdir(RESULT_DIR)
    sly.logger.info("Cityscapes Dataset folder has been created")

    class_to_id = []
    name2id = {}
    for idx, obj_class in enumerate(meta.obj_classes):
        if obj_class.geometry_type not in possible_geometries:
            continue
        curr_class = {}
        curr_class["name"] = obj_class.name
        curr_class["id"] = idx + 1
        curr_class["color"] = obj_class.color
        class_to_id.append(curr_class)
        name2id[obj_class.name] = (idx + 1, idx + 1, idx + 1)

    dump_json_file(class_to_id, os.path.join(RESULT_DIR, "class_to_id.json"))
    sly.logger.info("Writing classes with colors to class_to_id.json file")

    with tqdm(desc="Convert dataset to original format", total=project_fs.total_items) as pbar:
        for dataset in datasets:
            images_dir_path_train = os.path.join(result_images_train, dataset.name)
            images_dir_path_val = os.path.join(result_images_val, dataset.name)
            images_dir_path_test = os.path.join(result_images_test, dataset.name)
            anns_dir_path_train = os.path.join(result_anns_train, dataset.name)
            anns_dir_path_val = os.path.join(result_anns_val, dataset.name)
            anns_dir_path_test = os.path.join(result_anns_test, dataset.name)

            images = os.listdir(dataset.img_dir)

            if len(images) < 3:
                sly.logger.warn(
                    "Number of images in {} dataset is less then 3, val and train directories for this dataset will not be created".format(
                        dataset.name
                    )
                )

            # image_ids = [image_info.id for image_info in images]
            base_image_names = [sly.fs.get_file_name_with_ext(img) for img in images]
            # image_names = [
            #     get_file_name(image_info.name) + cityscapes_images_suffix + get_file_ext(image_info.name) for
            #     image_info in images
            # ]

            image_names = [
                get_file_name(img.replace("_leftImg8bit", ""))
                + cityscapes_images_suffix
                + get_file_ext(img)
                for img in images
            ]

            anns = [dataset.get_ann(name, meta) for name in image_names]

            splitter = get_tags_splitter(anns)
            curr_splitter = {"train": 0, "val": 0, "test": 0}

            for ann, image_name, base_image_name in zip(anns, image_names, base_image_names):
                train_val_flag = True
                try:
                    split_name = ann.img_tags.get("split").value
                    if split_name == "train":
                        image_dir_path = images_dir_path_train
                        ann_dir = anns_dir_path_train
                    elif split_name == "val":
                        image_dir_path = images_dir_path_val
                        ann_dir = anns_dir_path_val
                    else:
                        image_dir_path = images_dir_path_test
                        ann_dir = anns_dir_path_test
                        train_val_flag = False
                except:
                    ann_tags = [tag.name for tag in ann.img_tags]
                    separator_tags = list(set(ann_tags) & set(possible_tags))
                    if len(separator_tags) > 1:
                        sly.logger.warn(
                            """There are more then one separator tag for {} image. {}
                        tag will be used for split""".format(
                                image_name, separator_tags[0]
                            )
                        )

                    if len(separator_tags) >= 1:
                        if separator_tags[0] == "train":
                            image_dir_path = images_dir_path_train
                            ann_dir = anns_dir_path_train
                        elif separator_tags[0] == "val":
                            image_dir_path = images_dir_path_val
                            ann_dir = anns_dir_path_val
                        else:
                            image_dir_path = images_dir_path_test
                            ann_dir = anns_dir_path_test
                            train_val_flag = False

                    if len(separator_tags) == 0:
                        if curr_splitter["test"] == splitter["test"]:
                            curr_splitter = {"train": 0, "val": 0, "test": 0}
                        if curr_splitter["train"] < splitter["train"]:
                            curr_splitter["train"] += 1
                            image_dir_path = images_dir_path_train
                            ann_dir = anns_dir_path_train
                        elif curr_splitter["val"] < splitter["val"]:
                            curr_splitter["val"] += 1
                            image_dir_path = images_dir_path_val
                            ann_dir = anns_dir_path_val
                        elif curr_splitter["test"] < splitter["test"]:
                            curr_splitter["test"] += 1
                            image_dir_path = images_dir_path_test
                            ann_dir = anns_dir_path_test
                            train_val_flag = False

                get_image_and_ann()
                pbar.update(1)

    if to_format in ("tar", "both"):
        sly.fs.archive_directory(RESULT_DIR, RESULT_ARCHIVE)
        sly.logger.info("Result directory is archived")
        if to_format == "tar":
            shutil.rmtree(RESULT_DIR)

    if to_format == "tar":
        return RESULT_ARCHIVE
    elif to_format == "dir":
        return RESULT_DIR
    else:
        return (RESULT_ARCHIVE, RESULT_DIR)


def get_tags_splitter(anns):
    anns_without_possible_tags = 0
    for ann in anns:
        ann_tags = [tag.name for tag in ann.img_tags]
        separator_tags = list(set(ann_tags) & set(possible_tags))
        if len(separator_tags) == 0:
            anns_without_possible_tags += 1
    train_tags_cnt = round(anns_without_possible_tags * splitter_coef)
    val_tags_cnt = round((anns_without_possible_tags - train_tags_cnt) / 2)
    test_tags_cnt = anns_without_possible_tags - train_tags_cnt - val_tags_cnt
    return {"train": train_tags_cnt, "val": val_tags_cnt, "test": test_tags_cnt}


def image_ext_to_png(im_path):
    if get_file_ext(im_path) != ".png":
        im = Image.open(im_path).convert("RGB")
        im.save(im_path[: -1 * len(get_file_ext(im_path))] + ".png")
        silent_remove(im_path)


def from_ann_to_cityscapes_mask(ann, name2id, app_logger, train_val_flag):
    mask_color = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    mask_label = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    poly_json = {"imgHeight": ann.img_size[0], "imgWidth": ann.img_size[1], "objects": []}

    for label in ann.labels:
        if type(label.geometry) not in possible_geometries:
            continue
        if train_val_flag:
            label.geometry.draw(mask_color, label.obj_class.color)
        label.geometry.draw(mask_label, name2id[label.obj_class.name])
        if type(label.geometry) == Bitmap:
            curr_cnt = label.geometry.to_contours()
            if len(curr_cnt) == 0:
                continue
            elif len(curr_cnt) == 1:
                poly_for_contours = curr_cnt[0]
            else:
                for poly in curr_cnt:
                    cur_contours = poly.exterior_np.tolist()
                    if len(poly.interior) > 0 and label.obj_class.name != "out of roi":
                        app_logger.info(
                            "Labeled objects must never have holes in cityscapes format, existing holes will be sketched"
                        )
                    cityscapes_contours = list(map(lambda cnt: cnt[::-1], cur_contours))
                    poly_json["objects"].append(
                        {"label": label.obj_class.name, "polygon": cityscapes_contours}
                    )
                continue
        else:
            poly_for_contours = label.geometry

        if len(poly_for_contours.interior) > 0 and label.obj_class.name != "out of roi":
            app_logger.info(
                "Labeled objects must never have holes in cityscapes format, existing holes will be sketched"
            )

        contours = poly_for_contours.exterior_np.tolist()

        if label.obj_class.name == "out of roi":
            for curr_interior in poly_for_contours.interior_np:
                contours.append(poly_for_contours.exterior_np.tolist()[0])
                contours.extend(curr_interior.tolist())
                contours.append(curr_interior.tolist()[0])

        cityscapes_contours = list(map(lambda cnt: cnt[::-1], contours))
        poly_json["objects"].append({"label": label.obj_class.name, "polygon": cityscapes_contours})

    return mask_color, mask_label, poly_json

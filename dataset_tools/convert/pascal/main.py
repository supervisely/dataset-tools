import os
import json

from collections import OrderedDict
from shutil import copyfile

from dataset_tools.convert import unpack_if_archive

import numpy as np
import lxml.etree as ET
from PIL import Image
import supervisely as sly
from supervisely.io.fs import get_file_name
from supervisely.imaging.color import generate_rgb


default_classes_colors = {
    "neutral": (224, 224, 192),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bottle": (128, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "chair": (192, 0, 0),
    "cow": (64, 128, 0),
    "diningtable": (192, 128, 0),
    "dog": (64, 0, 128),
    "horse": (192, 0, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 128, 128),
    "pottedplant": (0, 64, 0),
    "sheep": (128, 64, 0),
    "sofa": (0, 192, 0),
    "train": (128, 192, 0),
    "tvmonitor": (0, 64, 128),
}

MASKS_EXTENSION = ".png"


def to_supervisely(input_path: str, output_path: str = None):
    input_path = unpack_if_archive(input_path)
    # Specific directory that must exist in input path.
    DIR_NAME = "VOCdevkit"

    dataset_dir = os.path.join(input_path, DIR_NAME)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Input directory must contain {DIR_NAME} directory, but it is not found in {input_path}."
        )

    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path), "PASCAL_TO_SLY")

    # Rename VOC to VOC2012 if necessary.
    if not os.path.isdir(os.path.join(dataset_dir, "VOC2012")):
        try:
            os.rename(os.path.join(dataset_dir, "VOC"), os.path.join(dataset_dir, "VOC2012"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Input directory must contain {DIR_NAME}/VOC or {DIR_NAME}/VOC2012 directory, "
                f"but it is not found in {input_path}."
            )

    lists_dir = os.path.join(dataset_dir, "VOC2012", "ImageSets", "Segmentation")
    imgs_dir = os.path.join(dataset_dir, "VOC2012", "JPEGImages")
    segm_dir = os.path.join(dataset_dir, "VOC2012", "SegmentationClass")
    inst_dir = os.path.join(dataset_dir, "VOC2012", "SegmentationObject")
    colors_file = os.path.join(dataset_dir, "VOC2012", "colors.txt")
    with_instances = os.path.isdir(inst_dir)

    obj_classes = sly.ObjClassCollection()

    src_datasets = {}
    if not os.path.isdir(lists_dir):
        raise RuntimeError(f"There is no directory {lists_dir}, but it is necessary")

    for filename in os.listdir(lists_dir):
        if filename.endswith(".txt"):
            ds_name = os.path.splitext(filename)[0]
            file_path = os.path.join(lists_dir, filename)
            sample_names = list(filter(None, map(str.strip, open(file_path, "r").readlines())))
            src_datasets[ds_name] = sample_names

    if os.path.isfile(colors_file):
        in_lines = filter(None, map(str.strip, open(colors_file, "r").readlines()))
        in_splitted = (x.split() for x in in_lines)
        cls2col = {x[0]: (int(x[1]), int(x[2]), int(x[3])) for x in in_splitted}
    else:
        cls2col = default_classes_colors

    obj_classes_list = [
        sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=color)
        for class_name, color in cls2col.items()
    ]

    obj_classes = obj_classes.add_items(obj_classes_list)
    color2class_name = {v: k for k, v in cls2col.items()}

    out_project = sly.Project(output_path, sly.OpenMode.CREATE)
    images_filenames = {}

    for image_path in sly.fs.list_files(imgs_dir):
        image_name_noext = sly.fs.get_file_name(image_path)
        if image_name_noext in images_filenames:
            raise RuntimeError(
                "Multiple image with the same base name {!r} exist.".format(image_name_noext)
            )
        images_filenames[image_name_noext] = image_path

    for ds_name, sample_names in src_datasets.items():
        if len(sample_names) == 0:
            continue
        ds = out_project.create_dataset(ds_name)
        percent_counter = 0

        for sample_name in sample_names:
            percent_counter += 1
            try:
                src_img_path = images_filenames[get_file_name(sample_name)]
            except Exception:
                src_img_path = images_filenames[sample_name]
            src_img_filename = os.path.basename(src_img_path)
            segm_path = os.path.join(segm_dir, sample_name + MASKS_EXTENSION)

            inst_path = None
            if with_instances:
                inst_path = os.path.join(inst_dir, sample_name + MASKS_EXTENSION)

            if all((x is None) or os.path.isfile(x) for x in [src_img_path, segm_path, inst_path]):
                try:
                    ann = get_ann(src_img_path, segm_path, inst_path, color2class_name)
                    ds.add_item_file(src_img_filename, src_img_path, ann=ann)
                except Exception as e:
                    exc_str = str(e)
                    sly.logger.warn(
                        f"Input sample skipped due to error: {exc_str}",
                        exc_info=True,
                        extra={
                            "exc_str": exc_str,
                            "dataset_name": ds_name,
                            "image": src_img_path,
                        },
                    )

            else:
                ds.add_item_file(src_img_filename, src_img_path, ann=None)

    out_meta = sly.ProjectMeta(obj_classes=obj_classes)
    out_project.set_meta(out_meta)

    return output_path


def get_ann(img_path, segm_path, inst_path, color2class_name):
    segmentation_img = sly.image.read(segm_path)

    if inst_path is not None:
        instance_img = sly.image.read(inst_path)
        colored_img = instance_img
        instance_img16 = instance_img.astype(np.uint16)
        col2coord = get_col2coord(instance_img16)
        curr_col2cls = (
            (col, color2class_name.get(tuple(segmentation_img[coord])))
            for col, coord in col2coord.items()
        )
        curr_col2cls = {
            k: v for k, v in curr_col2cls if v is not None
        }  # _instance_ color -> class name
    else:
        colored_img = segmentation_img
        segmentation_img = segmentation_img.astype(np.uint16)
        colors = list(get_col2coord(segmentation_img).keys())
        curr_col2cls = {curr_col: color2class_name[curr_col] for curr_col in colors}

    ann = sly.Annotation.from_img_path(img_path)

    for color, class_name in curr_col2cls.items():
        mask = np.all(colored_img == color, axis=2)  # exact match (3-channel img & rgb color)
        bitmap = sly.Bitmap(data=mask)
        obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap)

        ann = ann.add_label(sly.Label(bitmap, obj_class))
        #  clear used pixels in mask to check missing colors, see below
        colored_img[mask] = (0, 0, 0)

    if np.sum(colored_img) > 0:
        sly.logger.warn("Not all objects or classes are captured from source segmentation.")

    return ann


# returns mapping: (r, g, b) color -> some (row, col) for each unique color except black
def get_col2coord(img):
    img = img.astype(np.int32)
    h, w = img.shape[:2]
    colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
    unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
    indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
    return {
        (col // (256**2), (col // 256) % 256, col % 256): (indx // w, indx % w)
        for col, indx in col2indx.items()
        if col != 0
    }


###################
PASCAL_CONTOUR_THICKNESS = 3
TRAIN_VAL_SPLIT_COEF = 0.8
RESULT_SUBDIR_NAME = "VOCdevkit/VOC"
VALID_IMG_EXT = set([".jpe", ".jpeg", ".jpg"])

TRAIN_TAG_NAME = "train"
VAL_TAG_NAME = "val"
SPLIT_TAGS = set([TRAIN_TAG_NAME, VAL_TAG_NAME])

SUPPORTED_GEOMETRY_TYPES = set([sly.Bitmap, sly.Polygon])

images_dir_name = "JPEGImages"
ann_dir_name = "Annotations"
ann_class_dir_name = "SegmentationClass"
ann_obj_dir_name = "SegmentationObject"

trainval_sets_dir_name = "ImageSets"
trainval_sets_main_name = "Main"
trainval_sets_segm_name = "Segmentation"

pascal_contour_color = [224, 224, 192]
pascal_ann_ext = ".png"


def from_supervisely(input_path: str, output_path: str = None):
    input_path = unpack_if_archive(input_path)
    project_info = sly.Project(input_path, sly.OpenMode.READ)
    meta = project_info.meta

    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path), "SLY_TO_PASCAL")

    result_dir = output_path
    result_subdir = os.path.join(result_dir, RESULT_SUBDIR_NAME)

    result_ann_dir = os.path.join(result_subdir, ann_dir_name)
    result_images_dir = os.path.join(result_subdir, images_dir_name)
    result_class_dir_name = os.path.join(result_subdir, ann_class_dir_name)
    result_obj_dir = os.path.join(result_subdir, ann_obj_dir_name)
    result_imgsets_dir = os.path.join(result_subdir, trainval_sets_dir_name)

    sly.fs.mkdir(result_ann_dir)
    sly.fs.mkdir(result_imgsets_dir)
    sly.fs.mkdir(result_images_dir)
    sly.fs.mkdir(result_class_dir_name)
    sly.fs.mkdir(result_obj_dir)

    images_stats = []
    classes_colors = {}

    datasets = project_info.datasets
    dataset_names = ["trainval", "val", "train"]

    for dataset in datasets:
        if dataset.name in dataset_names:
            is_trainval = 1
        else:
            is_trainval = 0

        image_names = [f for f in os.listdir(dataset.img_dir)]
        src_image_paths = [os.path.join(dataset.img_dir, f) for f in image_names]
        anns_json = [
            json.load(open(os.path.join(dataset.ann_dir, f))) for f in os.listdir(dataset.ann_dir)
        ]
        dst_image_paths = [os.path.join(result_images_dir, f) for f in image_names]

        for src, dst in zip(src_image_paths, dst_image_paths):
            copyfile(src, dst)

        for image_name, image_path, ann_json in zip(image_names, dst_image_paths, anns_json):
            img_title, img_ext = os.path.splitext(image_name)
            cur_img_filename = image_name

            if is_trainval == 1:
                cur_img_stats = {"classes": set(), "dataset": dataset.name, "name": img_title}
                images_stats.append(cur_img_stats)
            else:
                cur_img_stats = {"classes": set(), "dataset": None, "name": img_title}
                images_stats.append(cur_img_stats)

            if img_ext not in VALID_IMG_EXT:
                orig_image_path = os.path.join(result_images_dir, cur_img_filename)

                jpg_image = img_title + ".jpg"
                jpg_image_path = os.path.join(result_images_dir, jpg_image)

                im = sly.image.read(orig_image_path)
                sly.image.write(jpg_image_path, im)
                sly.fs.silent_remove(orig_image_path)

            ann = sly.Annotation.from_json(ann_json, meta)
            tag = find_first_tag(ann.img_tags, SPLIT_TAGS)
            if tag is not None:
                cur_img_stats["dataset"] = tag.meta.name

            valid_labels = []
            for label in ann.labels:
                if type(label.geometry) in SUPPORTED_GEOMETRY_TYPES:
                    valid_labels.append(label)

            ann = ann.clone(labels=valid_labels)
            ann_to_xml(project_info, image_path, cur_img_filename, result_ann_dir, ann)
            for label in ann.labels:
                cur_img_stats["classes"].add(label.obj_class.name)
                classes_colors[label.obj_class.name] = tuple(label.obj_class.color)

            fake_contour_th = 0
            if PASCAL_CONTOUR_THICKNESS != 0:
                fake_contour_th = 2 * PASCAL_CONTOUR_THICKNESS + 1

            from_ann_to_instance_mask(
                ann,
                os.path.join(result_class_dir_name, img_title + pascal_ann_ext),
                fake_contour_th,
            )
            from_ann_to_class_mask(
                ann, os.path.join(result_obj_dir, img_title + pascal_ann_ext), fake_contour_th
            )

    classes_colors = OrderedDict((sorted(classes_colors.items(), key=lambda t: t[0])))

    with open(os.path.join(result_subdir, "colors.txt"), "w") as cc:
        if PASCAL_CONTOUR_THICKNESS != 0:
            cc.write(
                f"neutral {pascal_contour_color[0]} {pascal_contour_color[1]} {pascal_contour_color[2]}\n"
            )

        for k in classes_colors.keys():
            if k == "neutral":
                continue

            cc.write(f"{k} {classes_colors[k][0]} {classes_colors[k][1]} {classes_colors[k][2]}\n")

    imgs_to_split = [i for i in images_stats if i["dataset"] is None]
    train_len = int(len(imgs_to_split) * TRAIN_VAL_SPLIT_COEF)

    for img_stat in imgs_to_split[:train_len]:
        img_stat["dataset"] = TRAIN_TAG_NAME
    for img_stat in imgs_to_split[train_len:]:
        img_stat["dataset"] = VAL_TAG_NAME

    write_segm_set(is_trainval, images_stats, result_imgsets_dir)
    write_main_set(is_trainval, images_stats, meta, result_imgsets_dir)

    return output_path


def find_first_tag(img_tags, split_tags):
    for tag in split_tags:
        if img_tags.has_key(tag):
            return img_tags.get(tag)
    return None


def from_ann_to_instance_mask(ann, mask_outpath, contour_thickness):
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == "neutral":
            label.geometry.draw(mask, pascal_contour_color)
            continue

        label.geometry.draw_contour(mask, pascal_contour_color, contour_thickness)
        label.geometry.draw(mask, label.obj_class.color)

    im = Image.fromarray(mask)
    im = im.convert("P", palette=Image.ADAPTIVE)
    im.save(mask_outpath)


def from_ann_to_class_mask(ann, mask_outpath, contour_thickness):
    exist_colors = [[0, 0, 0], pascal_contour_color]
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == "neutral":
            label.geometry.draw(mask, pascal_contour_color)
            continue

        new_color = generate_rgb(exist_colors)
        exist_colors.append(new_color)
        label.geometry.draw_contour(mask, pascal_contour_color, contour_thickness)
        label.geometry.draw(mask, new_color)

    im = Image.fromarray(mask)
    im = im.convert("P", palette=Image.ADAPTIVE)
    im.save(mask_outpath)


def write_main_set(is_trainval, images_stats, meta_json, result_imgsets_dir):
    result_imgsets_main_subdir = os.path.join(result_imgsets_dir, trainval_sets_main_name)
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_main_subdir)

    res_files = ["trainval.txt", "train.txt", "val.txt"]
    for file in os.listdir(result_imgsets_segm_subdir):
        if file in res_files:
            copyfile(
                os.path.join(result_imgsets_segm_subdir, file),
                os.path.join(result_imgsets_main_subdir, file),
            )

    train_imgs = [i for i in images_stats if i["dataset"] == TRAIN_TAG_NAME]
    val_imgs = [i for i in images_stats if i["dataset"] == VAL_TAG_NAME]

    write_objs = [
        {"suffix": "trainval", "imgs": images_stats},
        {"suffix": "train", "imgs": train_imgs},
        {"suffix": "val", "imgs": val_imgs},
    ]

    if is_trainval == 1:
        trainval_imgs = [i for i in images_stats if i["dataset"] == TRAIN_TAG_NAME + VAL_TAG_NAME]
        write_objs[0] = {"suffix": "trainval", "imgs": trainval_imgs}

    for obj_cls in meta_json.obj_classes:
        if obj_cls.geometry_type not in SUPPORTED_GEOMETRY_TYPES:
            continue
        if obj_cls.name == "neutral":
            continue
        for o in write_objs:
            with open(
                os.path.join(result_imgsets_main_subdir, f'{obj_cls.name}_{o["suffix"]}.txt'), "w"
            ) as f:
                for img_stats in o["imgs"]:
                    v = "1" if obj_cls.name in img_stats["classes"] else "-1"
                    f.write(f'{img_stats["name"]} {v}\n')


def write_segm_set(is_trainval, images_stats, result_imgsets_dir):
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_segm_subdir)

    with open(os.path.join(result_imgsets_segm_subdir, "trainval.txt"), "w") as f:
        if is_trainval == 1:
            f.writelines(
                i["name"] + "\n"
                for i in images_stats
                if i["dataset"] == TRAIN_TAG_NAME + VAL_TAG_NAME
            )
        else:
            f.writelines(i["name"] + "\n" for i in images_stats)
    with open(os.path.join(result_imgsets_segm_subdir, "train.txt"), "w") as f:
        f.writelines(i["name"] + "\n" for i in images_stats if i["dataset"] == TRAIN_TAG_NAME)
    with open(os.path.join(result_imgsets_segm_subdir, "val.txt"), "w") as f:
        f.writelines(i["name"] + "\n" for i in images_stats if i["dataset"] == VAL_TAG_NAME)


def ann_to_xml(project_info, image_path, img_filename, result_ann_dir, ann):
    xml_root = ET.Element("annotation")

    ET.SubElement(xml_root, "folder").text = "VOC_" + project_info.name
    ET.SubElement(xml_root, "filename").text = img_filename

    xml_root_source = ET.SubElement(xml_root, "source")
    ET.SubElement(xml_root_source, "database").text = "Supervisely Project ID:" + str(
        project_info.name
    )
    ET.SubElement(xml_root_source, "annotation").text = "PASCAL VOC"
    # ET.SubElement(xml_root_source, "image").text = "Supervisely Image ID:" + str(image_info.id)
    ET.SubElement(xml_root_source, "image").text = "Supervisely Image ID:" + img_filename

    image = Image.open(image_path)
    width, height = image.size

    xml_root_size = ET.SubElement(xml_root, "size")
    # ET.SubElement(xml_root_size, "width").text = str(image_info.width)
    # ET.SubElement(xml_root_size, "height").text = str(image_info.height)
    ET.SubElement(xml_root_size, "width").text = str(width)
    ET.SubElement(xml_root_size, "height").text = str(height)
    ET.SubElement(xml_root_size, "depth").text = "3"

    ET.SubElement(xml_root, "segmented").text = "1" if len(ann.labels) > 0 else "0"

    for label in ann.labels:
        if label.obj_class.name == "neutral":
            continue

        bitmap_to_bbox = label.geometry.to_bbox()

        xml_ann_obj = ET.SubElement(xml_root, "object")
        ET.SubElement(xml_ann_obj, "name").text = label.obj_class.name
        ET.SubElement(xml_ann_obj, "pose").text = "Unspecified"
        ET.SubElement(xml_ann_obj, "truncated").text = "0"
        ET.SubElement(xml_ann_obj, "difficult").text = "0"

        xml_ann_obj_bndbox = ET.SubElement(xml_ann_obj, "bndbox")
        ET.SubElement(xml_ann_obj_bndbox, "xmin").text = str(bitmap_to_bbox.left)
        ET.SubElement(xml_ann_obj_bndbox, "ymin").text = str(bitmap_to_bbox.top)
        ET.SubElement(xml_ann_obj_bndbox, "xmax").text = str(bitmap_to_bbox.right)
        ET.SubElement(xml_ann_obj_bndbox, "ymax").text = str(bitmap_to_bbox.bottom)

    tree = ET.ElementTree(xml_root)

    img_name = os.path.join(result_ann_dir, os.path.splitext(img_filename)[0] + ".xml")
    ann_path = os.path.join(result_ann_dir, img_name)
    ET.indent(tree, space="    ")
    tree.write(ann_path, pretty_print=True)

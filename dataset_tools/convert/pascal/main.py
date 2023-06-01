import os

import numpy as np
import supervisely as sly
from supervisely.io.fs import get_file_name


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


def from_supervisely(input_path: str, output_path: str = None):
    raise NotImplementedError("Function is not implemented yet")


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

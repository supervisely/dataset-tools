

from typing import Literal

import os, zipfile
import supervisely as sly
import dataset_tools.convert.weed.sly_globals as g
from tqdm import tqdm
import gdown
from supervisely.io.fs import get_file_name
from supervisely.imaging.image import read
from cv2 import connectedComponents



#========================================================================================================

#========================================================================================================
# import cv2
# a = cv2.imread('/home/andrew/alex_work/temp/leafs_ds/temp/00000_labels:png.png')
# import numpy as np
# test = np.unique(a)
# cv2.imwrite('/home/andrew/alex_work/temp/leafs_ds/temp/testb.png', a * 40)
# a = sly.io.json.load_json_file('/home/andrew/alex_work/temp/leafs_ds/temp/ImgOldImgNew-validation-data/val/via_region_data.json')
# b = sly.io.json.load_json_file('/home/andrew/alex_work/temp/leafs_ds/temp/kinetics700_2020/train.json')
#

import scipy.io
# mat = scipy.io.loadmat('/home/andrew/alex_work/temp/leafs_ds/temp/full_dataset.mat')
a=0


def get_image_shape(img_path):
    im = read(img_path)

    return im.shape[0], im.shape[1]


def create_ann(img_path):
    labels = []

    height, width = get_image_shape(img_path)

    ann_name = get_file_name(img_path) + g.ann_suffix
    ann_path = os.path.join(g.annotations_path, ann_name)

    ann_mask = read(ann_path)[:, :, 0]

    for class_name, class_index in g.index_to_class.items():
        bool_mask = ann_mask == class_index
        ret, curr_mask = connectedComponents(bool_mask.astype('uint8'), connectivity=8)
        for i in range(1, ret):
            obj_mask = curr_mask == i
            if obj_mask.sum() < g.max_label_area:
                continue
            bitmap = sly.Bitmap(obj_mask)
            label = sly.Label(bitmap, g.meta.get_obj_class(class_name))
            labels.append(label)

    return sly.Annotation(img_size=(height, width), labels=labels)


def extract_zip():
    if zipfile.is_zipfile(g.archive_path):
        with zipfile.ZipFile(g.archive_path, 'r') as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        # g.my_app.stop()


def to_supervisely(api: sly.Api, WORKSPACE_ID):
    if not os.path.exists(g.archive_path):
        gdown.download(g.weed_url, g.archive_path, quiet=False)
    extract_zip()

    images_path = os.path.join(g.work_dir_path, g.folder_name, g.images_folder_name)
    g.annotations_path = os.path.join(g.work_dir_path, g.folder_name, g.annotation_folder_name)

    images_names = os.listdir(images_path)

    new_project = api.project.create(WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    new_dataset = api.dataset.create(new_project.id, g.dataset_name, change_name_if_conflict=True)

    with tqdm(desc="Upload items", total=len(images_names)) as pbar:
        for img_batch in sly.batched(images_names, batch_size=g.batch_size):

            img_pathes = [os.path.join(images_path, name) for name in img_batch]
            img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(img_path) for img_path in img_pathes]
            api.annotation.upload_anns(img_ids, anns)

            pbar.update(len(img_batch))

    return new_project.id


def from_supervisely(
    input_path: str, output_path: str = None, to_format: Literal["dir", "tar", "both"] = "both"
) -> str:
    pass
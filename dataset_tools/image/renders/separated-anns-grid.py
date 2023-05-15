import os
import random
from collections import defaultdict

import cv2
import numpy as np
from dotenv import load_dotenv

import supervisely as sly
from supervisely.imaging.font import get_readable_font_size
from tqdm import tqdm

# - [x] collect images infos and anns first from project ID or path
# - [x] if project path is given read images from path
#    else project ID is given download np images
# - [x] collect sample images for grid as np
# - [x] calculate shapes of small imgs (depends on rows and cols numbers)
# - [x] create empty img with result shape (16:9 or 9:16 - depends on cols/rows number)
# - [x] resize all images
# - [x] draw anns
# - [x] resize anns
# - [x] place imgs on grid


def get_sample(data, cnt):
    samples = []
    while len(samples) < cnt:
        item = random.choice(data)
        if item not in samples:
            samples.append(item)
    # random.shuffle(data)
    # data = data[:cnt]
    return samples


def calculate_shapes(max_image_size, w_ratio, h_ratio, rows, cols):
    height = width = max_image_size
    if rows > cols * 2:
        piece_h = height // rows
        piece_w = int(max(1, piece_h * w_ratio / h_ratio))
    else:
        piece_w = width // (cols * 2)
        piece_h = int(max(1, piece_w * h_ratio / w_ratio))
    height, width = piece_h * rows, piece_w * cols * 2

    sly.logger.info(f"Result image size is ({height}, {width})")
    return height, width, piece_h, piece_w


def resize_images(images, out_size):
    h, w = out_size
    resized_images = []

    for img in images:
        img_h, img_w = img.shape[:2]
        src_ratio = w / h
        img_ratio = img_w / img_h
        if img_ratio == src_ratio:
            res_img = sly.image.resize(img, (h, w))
        else:
            if img_ratio > src_ratio:
                img_h, img_w = int(w * img_ratio), w
                res_img = sly.image.resize(img, (img_h, img_w))
            else:
                img_h, img_w = h, int(h / img_ratio)
                res_img = sly.image.resize(img, (img_h, img_w))
            crop_rect = sly.Rectangle(
                top=(img_h - h) // 2,
                left=(img_w - w) // 2,
                bottom=(img_h + h) // 2,
                right=(img_w + w) // 2,
            )
            res_img = sly.image.crop_with_padding(res_img, crop_rect)

        resized_images.append(res_img)
    
    sly.logger.info(f"{len(resized_images)} images resized to {out_size}")
    return resized_images


def draw_masks_on_single_image(ann: sly.Annotation, image: sly.ImageInfo):
    height, width = image.height, image.width
    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    mask_img[:, :, :] = (0, 0, 0)

    for label in ann.labels:
        text = label.obj_class.name
        red, green, blue = sly.color.random_rgb()

        label.geometry.draw(mask_img, (red, green, blue), thickness=10)

        bbox_center = label.geometry.to_bbox().center
        col, row = bbox_center.col, bbox_center.row
        font_size = get_readable_font_size((width, height))
        font = sly.image.get_font(font_file_name=None, font_size=font_size)
        left, _, right, _ = font.getbbox(text)
        t_width = right - left

        def _put_text(org):
            cv2.putText(
                mask_img,
                text,
                org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=[0, 0, 0, 255],
                thickness=5,
                lineType=cv2.LINE_AA,
            )

        _put_text((int(col - t_width // 2 + 5), row + 3))
        _put_text((int(col - t_width // 2), row))

    return mask_img


# def get_grid_size(num: int = 1) -> tuple:
#     # get num of cols andd rows in result grid
#     cols = max(int(np.ceil(np.sqrt(num))), 1)
#     rows = max((num - 1) // cols + 1, 1)
#     return (rows, cols)


def create_image_grid(images, grid_size):
    img_h, img_w = images[0].shape[:2]
    num = len(images)
    grid_h, grid_w = grid_size

    grid = np.zeros(
        [grid_h * img_h, grid_w * img_w] + list(images[0].shape[-1:]),
        dtype=np.uint8,
    )
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w, ...] = images[idx]

    return grid


def main(api: sly.Api, project_id=None, project_dir=None, rows: int = 3, cols: int = 3):
    images_infos = []
    src_data = []
    ds_img_infos = defaultdict(list)
    ds_img_anns = defaultdict(list)
    ds_img_ids = defaultdict(list)
    np_images = []

    if project_id is not None:
        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        datasets = api.dataset.get_list(project_id)

        for dataset in datasets:
            images = api.image.get_list(dataset.id)
            images_infos.extend(images)

        src_data = get_sample(images_infos, rows * cols)

        with tqdm(desc="Collecting info from server", total=len(src_data)) as pbar:
            for image_info in src_data:
                ds_img_infos[image_info.dataset_id].append(image_info)
                ds_img_ids[image_info.dataset_id].append(image_info.id)

            for ds_id, imgs_ids in ds_img_ids.items():
                anns_json = api.annotation.download_json_batch(ds_id, imgs_ids)
                ds_img_anns[ds_id].extend(sly.Annotation.from_json(json, meta) for json in anns_json)

                np_images.extend(api.image.download_nps(ds_id, imgs_ids))
                pbar.update(len(imgs_ids))

    elif project_dir is not None:
        project_fs = sly.read_single_project(project_dir)
        meta = project_fs.meta

        datasets = project_fs.datasets

        for dataset in datasets:
            dataset: sly.Dataset
            images = [
                {dataset.name: [dataset.get_image_info(sly.fs.get_file_name(img))]}
                for img in os.listdir(dataset.ann_dir)
            ]
            images_infos.extend(images)

        src_data = get_sample(images_infos)

        with tqdm(desc="Collecting info from directory", total=len(src_data)) as pbar:
            for ds_name, image_info in src_data:
                ds_img_infos[ds_name].append(image_info)
                for ds in datasets:
                    ds: sly.Dataset
                    if ds.name == ds_name:
                        ds_img_anns[ds_name].extend(ds.get_ann(image_info.name, meta))
                        np_images.append(cv2.imread(ds.get_img_path(image_info.name)))
                        pbar.update(1)

    sly.logger.info(f"Source data for {len(src_data)} images is collected")

    height, width, piece_h, piece_w = calculate_shapes(1920, 16, 9, 3, 3)

    original_masks = []
    for anns, infos in zip(ds_img_anns.values(), ds_img_infos.values()):
        for ann, img_info in zip(anns, infos):
            original_masks.append(draw_masks_on_single_image(ann, img_info))

    resized_images = resize_images(np_images, (piece_h, piece_w))
    resized_masks = resize_images(original_masks, (piece_h, piece_w))

    img = create_image_grid(
        [i for pair in zip(resized_images, resized_masks) for i in pair],
        (height, width),
    )

    storage_dir = sly.app.get_data_dir()
    sly.fs.clean_dir(storage_dir)
    save_path = os.path.join(storage_dir, "separated_images_grid.jpg")
    # cv2.imwrite(save_path, img)
    sly.image.write(save_path, img)
    sly.logger.info(f"Result grid saved to: {save_path}")


if sly.is_development():
    load_dotenv("local.env")

project_id = sly.env.project_id(raise_not_found=False)

api = sly.Api()
main(api, project_id)

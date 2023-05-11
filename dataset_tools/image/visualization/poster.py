import os
import random
from collections import defaultdict
from typing import List

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image as PILImage
from PIL import ImageDraw

import supervisely as sly
from supervisely.imaging import font as sly_font


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

storage_dir = sly.app.get_data_dir()
sly.fs.clean_dir(storage_dir)

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
GAP = 20
IMG_WIDTH_1 = (IMAGE_WIDTH - 4 * GAP) // 3
IMG_HEIGHT_1 = (IMAGE_HEIGHT - 3 * GAP) // 5 * 3
IMG_WIDTH_2 = (IMAGE_WIDTH - 5 * GAP) // 4
IMG_HEIGHT_2 = (IMAGE_HEIGHT - 3 * GAP) // 5 * 2

BORDER_OFFSET = 4


api = sly.api.Api()
project_id = sly.env.project_id()
dataset_id = sly.env.dataset_id(raise_not_found=False)
project = api.project.get_info_by_id(project_id)
datasets = api.dataset.get_list(project.id)

project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

TEXT = (
    project.name.upper()
    if len(project.name.split(" ")) < 4
    else " ".join(project.name.split(" ")[:3])
)
TEXT_2 = f"{project.items_count} IMAGES"
TEXT_3 = f"{len(project_meta.obj_classes)} CLASSES"

labels_cnt = 0
for ds in datasets:
    image_infos = api.image.get_list(ds.id)
    for img_inf in image_infos:
        labels_cnt += img_inf.labels_count
TEXT_4 = f"{labels_cnt} LABELS"


def draw_text(image, anchor_point, text, font, is_title: bool = False):
    text_w, text_h = font.getsize(text)
    # prepare custom text bg
    top_left = (anchor_point[1] - BORDER_OFFSET, anchor_point[0])
    bottom_right = (
        anchor_point[1] + text_w + BORDER_OFFSET,
        anchor_point[0] + text_h + int(1.5 * BORDER_OFFSET),
    )

    image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = 0
    image[:, :, 3] = 255
    if is_title:
        top_left = (top_left[0] + 2, top_left[1] + 2)
        bottom_right = (bottom_right[0] - BORDER_OFFSET, bottom_right[1] - BORDER_OFFSET)
        image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = 255

    source_img = PILImage.fromarray(image)

    canvas = PILImage.new("RGBA", source_img.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(canvas, "RGBA")
    rect_top, rect_left = anchor_point

    if is_title:
        drawer.text((rect_left + 1, rect_top), text, fill=(0, 0, 0, 255), font=font)
    else:
        drawer.text((rect_left + 1, rect_top), text, font=font)

    source_img = PILImage.alpha_composite(source_img, canvas)
    image[:, :, :] = np.array(source_img, dtype=np.uint8)

    return image


def draw_all_text(image):
    font_ratio_main = 4
    font_ratio_2 = 1.5
    font_ratio_3 = 1.3
    font_ratio_4 = 1

    base_font_size = sly_font.get_readable_font_size([IMAGE_HEIGHT, IMAGE_WIDTH])

    # main title
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_main))
    left, top, right, bottom = font.getbbox(TEXT)
    text_w, text_h = right - left, bottom - top
    while text_w > IMAGE_WIDTH * 0.9:
        font_ratio_main = 0.9 * font_ratio_main
        font = font.font_variant(size=int(base_font_size * font_ratio_main))
        left, top, right, bottom = font.getbbox(TEXT)
        text_w, text_h = right - left, bottom - top
    main_anchor_point = ((IMAGE_HEIGHT - text_h) // 2, (IMAGE_WIDTH - text_w) // 2)
    image = draw_text(image, main_anchor_point, TEXT, font, is_title=True)

    # images count
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_2))
    left, top, right, bottom = font.getbbox(TEXT_2)
    text_w_2, text_h_2 = right - left, bottom - top
    anchor_point = (main_anchor_point[0] - text_h_2 - BORDER_OFFSET * 3, main_anchor_point[1])
    image = draw_text(image, anchor_point, TEXT_2, font)

    # classes count
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_3))
    left, top, right, bottom = font.getbbox(TEXT_3)
    text_w_3, text_h_3 = right, bottom
    anchor_point = (main_anchor_point[0] + text_h + 4 * BORDER_OFFSET, IMAGE_WIDTH // 2 - text_w_3)
    image = draw_text(image, anchor_point, TEXT_3, font)

    # masks count
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_4))
    left, top, right, bottom = font.getbbox(TEXT_4)
    text_w_4, text_h_4 = right - left, bottom - top
    anchor_point = (
        main_anchor_point[0] - text_h_4 - BORDER_OFFSET * 2,
        main_anchor_point[1] + text_w - text_w_4,
    )
    image = draw_text(image, anchor_point, TEXT_4, font)

    return image


def get_images_from_project(datasets):
    all_images_infos = []
    selected_image_infos = []

    if dataset_id is not None:
        datasets = [api.dataset.get_info_by_id(dataset_id)]

    for ds in datasets:
        all_images_infos.extend(api.image.get_list(ds.id))

    if len(all_images_infos) < 15:
        return all_images_infos

    while len(selected_image_infos) < 15:
        random.shuffle(all_images_infos)
        selected_image_infos.append(all_images_infos[0])

    return selected_image_infos


def download_selected_images(selected_image_infos: List[sly.ImageInfo]):
    np_images = []
    anns = []
    temp_images = []
    temp_anns = []
    ids2ds = defaultdict(list)

    for img in selected_image_infos:
        ids2ds[img.dataset_id].append(img.id)
    for ds_id, img_ids in ids2ds.items():
        temp_images.extend(api.image.download_nps(ds_id, img_ids))
        anns_json = api.annotation.download_json_batch(ds_id, img_ids)
        temp_anns.extend(
            [sly.Annotation.from_json(ann_json, project_meta) for ann_json in anns_json]
        )

    i = 0
    while len(np_images) < 7:
        temp_anns: List[sly.Annotation]
        if len(temp_anns[i].labels) < 1:
            continue
        np_images.append(temp_images[i])
        anns.append(temp_anns[i])
        i += 1

    for img, ann in zip(np_images, anns):
        ann: sly.Annotation
        ann.draw_pretty(img, thickness=2, opacity=0.3)

    np_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in np_images]
    for i, img in enumerate(np_images[:3]):
        np_images[i] = sly.image.resize_inter_nearest(img, (IMG_HEIGHT_1, IMG_WIDTH_1))
    for i, img in enumerate(np_images[3:]):
        np_images[i + 3] = sly.image.resize_inter_nearest(img, (IMG_HEIGHT_2, IMG_WIDTH_2))
    for i, img in enumerate(np_images):
        rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        if i % 2 == 0:
            background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            background[:, :, :3] = 255
            alpha = 0.4
            img = cv2.addWeighted(background, 1 - alpha, img, alpha, 0)
        rgba_image[:, :, 3] = 255
        rgba_image[:, :, :3] = img
        np_images[i] = rgba_image
    return np_images


def create_frame(background, images: List[np.ndarray], gap):
    x, y = gap, gap
    for img in images[:3]:
        x_end, y_end = x + img.shape[1], y + img.shape[0]
        background[y:y_end, x:x_end] = img
        x = x_end + gap
    y = y_end + gap
    x = gap

    for img in images[3:]:
        x_end, y_end = x + img.shape[1], y + img.shape[0]
        background[y:y_end, x:x_end] = img
        x = x_end + gap
    return background


# empty image with white bg
image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.uint8)
image[:, :, 0:3] = 255
image[:, :, 3] = 255


selected_image_infos = get_images_from_project(datasets)
selected_image_nps = download_selected_images(selected_image_infos)
image = create_frame(image, selected_image_nps, GAP)

image = draw_all_text(image)

save_path = os.path.join(storage_dir, "image.png")
# save image
cv2.imwrite(save_path, image)


# sly.fs.remove_dir(storage_dir)

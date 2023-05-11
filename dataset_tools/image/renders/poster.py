import os
import random
from collections import defaultdict
from typing import List

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image as PILImage
from PIL import ImageDraw
from tqdm import tqdm

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


def draw_text(image, anchor_point, text, font, is_title: bool = False):
    left, top, right, bottom = font.getbbox(text)
    text_w, text_h = right - left, bottom - top
    # prepare custom text bg
    top_left = (anchor_point[1] - BORDER_OFFSET, anchor_point[0])
    bottom_right = (
        anchor_point[1] + text_w + BORDER_OFFSET,
        anchor_point[0] + text_h + int(1.5 * BORDER_OFFSET + top),
    )

    image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = 0
    image[:, :, 3] = 255
    if is_title:
        top_left = (top_left[0] + BORDER_OFFSET, top_left[1] + BORDER_OFFSET)
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


def draw_all_text(image, title, img_cnt_text, cls_cnt_text, masks_cnt_text):
    sly.logger.info(f"Start drawing texts on poster.")
    font_ratio_main = 4
    font_ratio_2 = 1.5
    font_ratio_3 = 1.3
    font_ratio_4 = 1

    base_font_size = sly_font.get_readable_font_size([IMAGE_HEIGHT, IMAGE_WIDTH])

    # main title
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_main))
    left, top, right, bottom = font.getbbox(title)
    text_w, text_h = right - left, bottom - top
    while text_w > IMAGE_WIDTH * 0.9:
        font_ratio_main = 0.9 * font_ratio_main
        font = font.font_variant(size=int(base_font_size * font_ratio_main))
        left, top, right, bottom = font.getbbox(title)
        text_w, text_h = right - left, bottom - top
    main_anchor_point = ((IMAGE_HEIGHT - text_h) // 2, (IMAGE_WIDTH - text_w) // 2)
    image = draw_text(image, main_anchor_point, title, font, is_title=True)

    # images count
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_2))
    left, top, right, bottom = font.getbbox(img_cnt_text)
    text_w_2, text_h_2 = right - left, bottom - top
    anchor_point = (main_anchor_point[0] - text_h_2 - BORDER_OFFSET * 3, main_anchor_point[1])
    image = draw_text(image, anchor_point, img_cnt_text, font)

    # classes count
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_3))
    left, top, right, bottom = font.getbbox(cls_cnt_text)
    text_w_3, text_h_3 = right, bottom
    anchor_point = (main_anchor_point[0] + text_h + 4 * BORDER_OFFSET, IMAGE_WIDTH // 2 - text_w_3)
    image = draw_text(image, anchor_point, cls_cnt_text, font)

    # masks count
    font = sly_font.get_font(font_size=int(base_font_size * font_ratio_4))
    left, top, right, bottom = font.getbbox(masks_cnt_text)
    text_w_4, text_h_4 = right - left, bottom - top
    anchor_point = (
        main_anchor_point[0] - text_h_4 - BORDER_OFFSET * 2,
        main_anchor_point[1] + text_w - text_w_4,
    )
    image = draw_text(image, anchor_point, masks_cnt_text, font)

    sly.logger.info(f"Drawing texts on poster successfully finished.")
    return image


def download_selected_images(selected_image_infos: List[sly.ImageInfo], project_meta):
    sly.logger.info(f"Downloading 7 sample images.")
    np_images = []

    i = 0
    with tqdm(desc="Downloading 7 sample images.", total=7) as pbar:
        while len(np_images) < 7:
            img_info = selected_image_infos[i]
            ann_json = api.annotation.download_json(img_info.id)
            ann = sly.Annotation.from_json(ann_json, project_meta)
            if len(ann.labels) < 1:
                i += 1
                continue
            np_img = api.image.download_np(img_info.id)
            ann.draw_pretty(np_img, thickness=2, opacity=(1 - (i + 3) / 10))
            np_images.append(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
            i += 1
            pbar.update(1)

    sly.logger.info(f"Resizing 7 sample images.")
    for i, img in enumerate(np_images):
        if i < 3:
            h, w = IMG_HEIGHT_1, IMG_WIDTH_1
        else:
            h, w = IMG_HEIGHT_2, IMG_WIDTH_2
        img_h, img_w = img.shape[1], img.shape[0]
        src_ratio = w / h
        save_path = os.path.join(storage_dir, "poster.png")
        # save image
        cv2.imwrite(save_path, img)
        img_ratio = img_w / img_h
        if img_ratio != src_ratio:
            if img_ratio > src_ratio:
                img_h, img_w = int(w * img_ratio), w
                img = sly.image.resize(img, (img_h, img_w))
            else:
                img_h, img_w = h, int(h / img_ratio)
                img = sly.image.resize(img, (img_h, img_w))
            save_path = os.path.join(storage_dir, "poster.png")
            # save image
            cv2.imwrite(save_path, img)
            crop_rect = sly.Rectangle(
                top=(img_h - h) // 2,
                left=(img_w - w) // 2,
                bottom=(img_h + h) // 2,
                right=(img_w + w) // 2,
            )
            img = sly.image.crop_with_padding(img, crop_rect)
            save_path = os.path.join(storage_dir, "poster.png")
            # save image
            cv2.imwrite(save_path, img)
        else:
            img = sly.image.resize(img, (h, w))

        rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

        if i % 2 == 0:
            background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            background[:, :, :3] = 255
            alpha = 0.4
            img = cv2.addWeighted(background, 1 - alpha, img, alpha, 0)
        rgba_image[:, :, 3] = 255
        rgba_image[:, :, :3] = img
        np_images[i] = rgba_image
    sly.logger.info(f"Successfully resized 7 sample images.")
    return np_images


def create_frame(background, images: List[np.ndarray], gap):
    sly.logger.info(f"Start creating frame.")
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
    sly.logger.info(f"Frame successfully created.")
    return background


def create_poster(project_id=None, project_path=None):
    images_infos = []
    if project_id is not None:
        project = api.project.get_info_by_id(project_id)
        project_meta_json = api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta_json)

        datasets = api.dataset.get_list(project.id)
        if dataset_id is not None:
            datasets = [api.dataset.get_info_by_id(dataset_id)]

        for ds in datasets:
            images_infos.extend(api.image.get_list(ds.id))

    else:
        project = sly.Project(project_path, sly.OpenMode.READ)
        project_meta = project.meta
        datasets = project.datasets
        for dataset in datasets:
            dataset: sly.Dataset
            images_infos.extend(
                dataset.get_image_info(sly.fs.get_file_name(img))
                for img in os.listdir(dataset.ann_dir)
            )

    random.shuffle(images_infos)
    labels_cnt = 0
    for img_inf in images_infos:
        labels_cnt += img_inf.labels_count

    TEXT = (
        project.name.upper()
        if len(project.name.split(" ")) < 4
        else " ".join(project.name.split(" ")[:3])
    )
    TEXT_2 = (
        f"{project.items_count} IMAGES"
        if project_id is not None
        else f"{project.total_items} IMAGES"
    )
    TEXT_3 = f"{len(project_meta.obj_classes)} CLASSES"
    TEXT_4 = f"{labels_cnt} LABELS"

    # empty image with white bg
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.uint8)
    image[:, :, 0:3] = 255
    image[:, :, 3] = 255

    selected_image_nps = download_selected_images(images_infos, project_meta)
    image = create_frame(image, selected_image_nps, GAP)

    image = draw_all_text(image, TEXT, TEXT_2, TEXT_3, TEXT_4)

    save_path = os.path.join(storage_dir, "poster.png")
    # save image
    cv2.imwrite(save_path, image)

    # upload stats to Team files
    dst_path = f"/renders/{project.name}/poster.png"
    file_info = api.file.upload(team_id, save_path, dst_path)
    sly.logger.info(f"Dataset poster uploaded to Team files path: {file_info.path}")

    return file_info.path


team_id = sly.env.team_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

################## Option 1. Get poster for project with given ID ##################
upload_path = create_poster(project_id=project_id)


###################################### or #####################################

################# Option 2. Get poster for project with given path #################
# n_count = api.project.get_info_by_id(project_id).items_count
# p = tqdm(desc="Downloading project", total=n_count)

# sly.download_project(
#     api,
#     project_id,
#     storage_dir,
#     progress_cb=p.update,
#     save_image_info=True,
#     save_images=False,
# )
# upload_path = create_poster(project_path=storage_dir)

print(f"Poster uploaded to Team files: {upload_path}")

sly.fs.remove_dir(storage_dir)

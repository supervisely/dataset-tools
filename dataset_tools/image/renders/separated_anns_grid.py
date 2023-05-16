import os
import random

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import supervisely as sly


class SideAnnotationsGrid:
    def __init__(self, project_meta: sly.ProjectMeta, api: sly.Api = None, rows: int = 3, cols: int = 3):
        self.project_meta = project_meta

        self._max_size = 1920
        self._rows = rows
        self._cols = cols
        self._aspect_ratio = 9 / 16

        height, width, piece_h, piece_w = self._calculate_shapes()
        self._grid_size = (height, width)
        self._piece_size = (piece_h, piece_w)

        self._all_image_infos = []
        self._all_anns = []
        self.np_images = []
        self.original_masks = []
        self._grid = None

        self._api = api if api is not None else sly.Api.from_env()

    def update(self, data: tuple):
        cnt = self._cols * self._rows
        temp_imgs = []
        temp_anns = []
        for dataset_id, image_infos, anns in data:
            temp_imgs.extend(image_infos)
            temp_anns.extend(anns)
        temp_data = list(zip(temp_imgs, temp_anns))

        random.shuffle(temp_data)
        with tqdm(desc="Downloading images", total=cnt) as p:
            for img_info, ann in temp_data[:cnt]:
                self._all_image_infos.append(img_info)
                self._all_anns.append(ann)
                self.np_images.append(self._api.image.download_np(img_info.id))
                p.update(1)

                self.original_masks.append(self._draw_masks_on_single_image(ann, img_info))

        resized_images = self._resize_images(self.np_images)
        resized_masks = self._resize_images(self.original_masks)

        img = self._create_image_grid(
            [i for pair in zip(resized_images, resized_masks) for i in pair]
        )
        self._grid = img

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            sly.fs.clean_dir(storage_dir)
            path = os.path.join(storage_dir, "separated_images_grid.jpeg")
        sly.image.write(path, self._grid)
        sly.logger.info(f"Result grid saved to: {path}")

    def _calculate_shapes(
        self,
    ):
        height = width = self._max_size
        if self._rows > self._cols * 2:
            piece_h = height // self._rows
            piece_w = int(max(1, piece_h / self._aspect_ratio))
        else:
            piece_w = width // (self._cols * 2)
            piece_h = int(max(1, piece_w * self._aspect_ratio))
        height, width = piece_h * self._rows, piece_w * self._cols * 2

        sly.logger.info(f"Result image size is ({height}, {width})")
        return height, width, piece_h, piece_w

    def _resize_images(self, images):
        h, w = self._piece_size
        resized_images = []

        for img in images:
            img_h, img_w = img.shape[:2]
            src_ratio = w / h
            img_ratio = img_w / img_h
            if img_ratio == src_ratio:
                img = sly.image.resize(img, (h, w))
            else:
                if img_ratio < src_ratio:
                    img_h, img_w = int((w * img_h) // img_w), w
                else:
                    img_h, img_w = h, int((h * img_w) // img_h)
                img = sly.image.resize(img, (img_h, img_w))
                crop_rect = sly.Rectangle(
                    top=(img_h - h) // 2,
                    left=(img_w - w) // 2,
                    bottom=(img_h + h) // 2,
                    right=(img_w + w) // 2,
                )
                img = sly.image.crop_with_padding(img, crop_rect)
                img = sly.image.resize(img, (h, w))

            resized_images.append(img)

        sly.logger.info(f"{len(resized_images)} images resized to {self._piece_size}")
        return resized_images

    def _draw_masks_on_single_image(self, ann: sly.Annotation, image: sly.ImageInfo):
        height, width = image.height, image.width
        mask_img = np.zeros((height, width, 3), dtype=np.uint8)
        mask_img[:, :, 0:3] = (165, 180, 180)  # rgb(165, 180, 180)

        for label in ann.labels:
            random_rgb = sly.color.random_rgb()
            label.geometry.draw(mask_img, random_rgb, thickness=3)

        return mask_img

    def _create_image_grid(self, images):
        img_h, img_w = images[0].shape[:2]
        num = len(images)
        grid_h, grid_w = self._grid_size

        grid = np.zeros(
            [grid_h, grid_w, 3],
            dtype=np.uint8,
        )
        for idx in range(num):
            x = (idx % (self._cols * 2)) * img_w
            y = (idx // (self._cols * 2)) * img_h
            grid[y : y + img_h, x : x + img_w, ...] = images[idx][:, :, ...]

        return grid


###########################################################################
############################# for local debug #############################
###########################################################################

# if sly.is_development():
#     load_dotenv(os.path.expanduser("~/ninja.env"))
#     load_dotenv("local.env")

# project_id = sly.env.project_id(raise_not_found=False)

# api = sly.Api.from_env()

# project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
# grid = SideAnnotationsGrid(project_meta=project_meta, rows=5, cols=2)

# limit = 30
# samples = []
# for dataset in api.dataset.get_list(project_id):
#     ds_images = api.image.get_list(
#         dataset.id,
#         filters=[{"field": "labelsCount", "operator": ">", "value": "0"}],
#     )
#     if len(ds_images) == 0:
#         continue
#     sample_image_infos = random.sample(ds_images, limit)
#     ds_anns_infos = api.annotation.download_json_batch(
#         dataset.id, [s.id for s in sample_image_infos]
#     )
#     ds_anns = [sly.Annotation.from_json(a, project_meta) for a in ds_anns_infos]
#     samples.append((dataset.id, sample_image_infos, ds_anns))

# # for dataset_id, ds_images, ds_anns in samples:
# grid.update(samples)

# grid.to_image("/tmp/grid.png")

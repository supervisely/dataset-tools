import os
import random
import numpy as np
from dotenv import load_dotenv

import supervisely as sly


if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()

project_id = sly.env.project_id()
project_info = api.project.get_info_by_id(project_id)
meta_json = api.project.get_meta(project_id)


def calculate_avg_img_size(datasets, imagesCount, max_imgs_for_average=30):
    sizes = []
    included_images = 0

    for ds_idx, dataset in enumerate(datasets):
        images = api.image.get_list(dataset.id)
        random.shuffle(images)
        if imagesCount > 0:
            if ds_idx >= len(datasets) - 1:
                img_cnt = imagesCount - included_images
                if included_images + img_cnt >= max_imgs_for_average:
                    img_cnt = max_imgs_for_average - included_images
                images = images[:img_cnt]
            else:
                img_cnt = len(images)
                if included_images + img_cnt >= max_imgs_for_average:
                    img_cnt = max_imgs_for_average - included_images
                images = images[:img_cnt]

        for _, item_name in enumerate(images):
            sizes.append((item_name.height, item_name.width))
            included_images += 1

    sizes = np.array(sizes)
    avg_img_size = (
        sizes[:, 0].mean().astype(np.int32).item(),
        sizes[:, 1].mean().astype(np.int32).item(),
    )
    return avg_img_size


def get_heatmap(api, class_name):
    imagesCount = project_info.items_count
    geometry_types_to_heatmap = ["polygon", "rectangle", "bitmap"]
    meta = sly.ProjectMeta.from_json(meta_json)

    datasets = api.dataset.get_list(project_info.id)

    avg_img_size = calculate_avg_img_size(datasets, imagesCount)

    heatmap = np.zeros(avg_img_size + (3,), dtype=np.float32)
    included_images = 0
    for ds_idx, dataset in enumerate(datasets):
        images = api.image.get_list(dataset.id)
        random.shuffle(images)
        if imagesCount > 0:
            if ds_idx >= len(datasets) - 1:
                images = images[: imagesCount - included_images]
            else:
                img_cnt = len(images)
                images = images[:img_cnt]

        for _, item_infos in enumerate(sly.batched(images)):
            img_ids = [x.id for x in item_infos]
            ann_infos = api.annotation.download_batch(dataset.id, img_ids)
            anns = [sly.Annotation.from_json(x.annotation, meta) for x in ann_infos]
            for ann in anns:
                ann = ann.resize(avg_img_size)
                temp_canvas = np.zeros(avg_img_size + (3,), dtype=np.uint8)
                for label in ann.labels:
                    if (
                        label.obj_class.name == class_name
                        and label.geometry.geometry_name() in geometry_types_to_heatmap
                    ):
                        ann = ann.delete_label(label)
                        label.draw(temp_canvas, color=(1, 1, 1))
                heatmap += temp_canvas

            included_images += len(item_infos)

    return heatmap


def create_heatmaps(api: sly.Api):
    heatmaps = {}
    classes = [obj_class.get("title") for obj_class in meta_json["classes"]]

    for class_name in classes:
        heatmap = get_heatmap(api, class_name)
        heatmaps[class_name] = heatmap

    return heatmaps


### For visual debug:
#
# import matplotlib
# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     heatmaps = create_heatmaps(api)
#     cmap = matplotlib.colormaps.get_cmap("viridis")
#     for heatmap in heatmaps:
#         plt.imshow(heatmaps[heatmap][:, :, 0], cmap=cmap)
#         plt.colorbar(cmap=cmap)
#         plt.show()

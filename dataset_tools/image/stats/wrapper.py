import random

from collections import defaultdict

import supervisely as sly


def sample_images(api, datasets, sample_rate):
    all_images = []
    for dataset in datasets:
        images = api.image.get_list(dataset.id)
        all_images.extend(images)

    cnt_images = len(all_images)
    if sample_rate != 1:
        cnt_images = int(max(1, sample_rate * len(all_images)))
        random.shuffle(all_images)
        all_images = all_images[:cnt_images]

    ds_images = defaultdict(list)
    for image_info in all_images:
        ds_images[image_info.dataset_id].append(image_info)
    return ds_images, cnt_images


def calculate(api, cfg=None, project_id=None, project_dir=None, sample_rate=1):
    result = {}

    # cfg = {
    #     #    "spatial": dtools.stat.spation_distribution,
    #     "classes": [
    #         classes_distribution,
    #         classes_cooccurence,
    #     ],
    #     #     "images": dtools.stat.classes-on-every-image,
    #     #    "objects": dtools.stat.classes-on-every-image,
    # }
    # supported_callbacks = [classes_distribution, classes_cooccurence]
    # send_callbacks = []

    # decomp_cfg = {}
    # for key, value in cfg.items():
    # if isinstance(value, (list, tuple)):
    #     for i, item in enumerate(value):
    #         new_key = f"{key}_{i+1}"
    #         decomp_cfg[new_key] = item
    # else:
    #     new_key = f"{key}_{1}"
    #     decomp_cfg[new_key] = value

    if project_id is not None:
        meta_json = api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        for statsType, Statistics in cfg.items():
            stats = {}
            datasets = api.dataset.get_list(project_id)
            ds_images, sample_count = sample_images(api, datasets, sample_rate)

            Statistics.prepare_data(stats, meta)

            for dataset_id, images in ds_images.items():
                dataset = api.dataset.get_info_by_id(dataset_id)
                # stats["currentDataset"] = dataset

                for img_batch in sly.batched(images):
                    image_ids = [image_info.id for image_info in img_batch]
                    ann_batch = api.annotation.download_batch(dataset.id, image_ids)

                    for image_info, ann_info in zip(img_batch, ann_batch):
                        # update_classes_distribution(stats, image, ann)

                        #  maybe *args **kwargs?
                        Statistics.update(stats, image_info, ann_info, meta, dataset)

            Statistics.aggregate_calculations(stats)

            result[statsType] = stats

    else:
        # TODO work with local sly format data
        pass

    return result

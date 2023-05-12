import io, json, os, random
from collections import defaultdict
from typing import List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import supervisely as sly


# if sly.is_development():
    

# api = sly.api.Api()

# PROJECT_ID = sly.env.project_id()
# TEAM_ID = sly.env.team_id()
# WORKSPACE_ID = sly.env.workspace_id()


# collect all table rows data
# def calculate_stats_per_image(
#     stats: dict,
#     images: List[sly.ImageInfo],
#     project_meta: sly.ProjectMeta,
#     class_names: List[str],
#     class_indices_colors: List[List[int]],
#     _name_to_index: dict,
#     ds_pbar: tqdm,
# ):
#     sum_class_area_per_image = [0] * len(class_names)
#     sum_class_count_per_image = [0] * len(class_names)
#     count_images_with_class = [0] * len(class_names)
#     dataset = api.dataset.get_info_by_id(images[0].dataset_id)

#     all_stats = []
#     for batch_images in sly.batched(images, batch_size=20):
#         img_ids = [img.id for img in batch_images]
#         anns = api.annotation.download_json_batch(batch_images[0].dataset_id, img_ids)
#         for img_info, ann_json in zip(batch_images, anns):
#             ann = sly.Annotation.from_json(ann_json, project_meta)
#             render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
#             render_idx_rgb[:] = BG_COLOR
#             ann.draw_class_idx_rgb(render_idx_rgb, _name_to_index)
#             stat_area = sly.Annotation.stat_area(render_idx_rgb, class_names, class_indices_colors)
#             stat_count = ann.stat_class_count(class_names)

#             if stat_area["unlabeled"] > 0:
#                 stat_count["unlabeled"] = 1

#             table_row = []
#             table_row.append(img_info.id)

#             table_row.append(
#                 '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(
#                     api.image.url(
#                         TEAM_ID, WORKSPACE_ID, dataset.project_id, dataset.id, img_info.id
#                     ),
#                     img_info.name,
#                 )
#             )

#             table_row.append(dataset.name)
#             area_unl = stat_area["unlabeled"] if not np.isnan(stat_area["unlabeled"]) else 0
#             table_row.extend(
#                 [
#                     stat_area["height"],
#                     stat_area["width"],
#                     stat_area["channels"],
#                     round(area_unl, 2),
#                 ]
#             )
#             for idx, class_name in enumerate(class_names):
#                 cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
#                 cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
#                 sum_class_area_per_image[idx] += cur_area
#                 sum_class_count_per_image[idx] += cur_count
#                 count_images_with_class[idx] += 1 if stat_count[class_name] > 0 else 0
#                 if class_name == "unlabeled":
#                     continue
#                 table_row.append(round(cur_area, 2))
#                 table_row.append(round(cur_count, 2))

#             if len(table_row) != len(stats["columns"]):
#                 raise RuntimeError("Values for some columns are missed")

#             all_stats.append(table_row)
#             ds_pbar.update(1)
#     stats["data"] = all_stats


# # prepare table columns
# def process_obj_classes(stats, project_meta):
#     class_names = ["unlabeled"]
#     class_colors = [[0, 0, 0]]
#     class_indices_colors = [[0, 0, 0]]
#     _name_to_index = {}
#     table_columns = ["image id", "image", "dataset", "height", "width", "channels", "unlabeled"]

#     for idx, obj_class in enumerate(project_meta.obj_classes):
#         class_names.append(obj_class.name)
#         class_colors.append(obj_class.color)
#         class_index = idx + 1
#         class_indices_colors.append([class_index, class_index, class_index])
#         _name_to_index[obj_class.name] = class_index
#         table_columns.append(get_col_name_area(obj_class.name, obj_class.color))
#         table_columns.append(get_col_name_count(obj_class.name, obj_class.color))
#     stats["columns"] = table_columns
#     return class_names, class_indices_colors, _name_to_index


# def project_per_image_stats(project_id: int = None, project_path: str = None, sample_procent=None):
#     stats = defaultdict(dict)

#     # if project_id is not None and project_path is not None:
#     #     raise Exception(
#     #         'Both "project_id" and "project_path" attributes are defined, but only one is allowed.'
#     #     )

#     # if project_id is None and project_path is None:
#     #     raise Exception('Both "project_id" and "project_path" attributes are not defined.')

#     if project_id is not None:
#         project_info = api.project.get_info_by_id(project_id)
#         if project_info is None:
#             raise Exception(f"There is not project with ID ({project_id}) in current workspace.")

#         project_meta_json = api.project.get_meta(project_id)
#         project_meta = sly.ProjectMeta.from_json(project_meta_json)

#         # class_names, class_indices_colors, _name_to_index = process_obj_classes(stats, project_meta)

#         total_items = project_info.items_count
#         if sample_procent is not None:
#             total_items = 0
#             for ds in api.dataset.get_list(project_id):
#                 total_items += int(max(1, ds.items_count * sample_procent // 100))
#         with tqdm(desc="Calculating stats using project ID", total=total_items) as ds_pbar:
#             datasets = api.dataset.get_list(project_id)
#             for dataset in datasets:
#                 images = api.image.get_list(dataset.id)
#                 images = (
#                     get_sample(images, sample_procent) if sample_procent is not None else images
#                 )

#                 calculate_stats_per_image(
#                     stats,
#                     images,
#                     project_meta,
#                     class_names,
#                     class_indices_colors,
#                     _name_to_index,
#                     ds_pbar,
#                 )

#     if project_path is not None:
#         project_fs = sly.Project(project_path, sly.OpenMode.READ)
#         project_meta = project_fs.meta

#         class_names, class_indices_colors, _name_to_index = process_obj_classes(stats, project_meta)
#         total_items = project_fs.total_items
#         if sample_procent is not None:
#             total_items = 0
#             for ds in project_fs.datasets:
#                 curr_cnt = len(os.listdir(ds.ann_dir))
#                 total_items += int(max(1, curr_cnt * sample_procent // 100))
#         with tqdm(desc="Calculating stats for local project", total=total_items) as ds_pbar:
#             datasets = project_fs.datasets
#             for dataset in datasets:
#                 dataset: sly.Dataset
#                 images = [
#                     dataset.get_image_info(sly.fs.get_file_name(img))
#                     for img in os.listdir(dataset.ann_dir)
#                 ]
#                 images = (
#                     get_sample(images, sample_procent) if sample_procent is not None else images
#                 )

#                 calculate_stats_per_image(
#                     stats,
#                     images,
#                     project_meta,
#                     class_names,
#                     class_indices_colors,
#                     _name_to_index,
#                     ds_pbar,
#                 )

#     return stats


# storage_dir = sly.app.get_data_dir()

# ################## Option 1. Get stats with given project ID ##################
# stats = project_per_image_stats(project_id=PROJECT_ID, sample_procent=5)

# ###################################### or #####################################

# ################# Option 2. Get stats with given project path #################
# n_count = api.project.get_info_by_id(PROJECT_ID).items_count
# p = tqdm(desc="Downloading", total=n_count)

# sly.download_project(
#     api,
#     PROJECT_ID,
#     storage_dir,
#     progress_cb=p.update,
#     save_image_info=True,
#     save_images=False,
# )
# stats = project_per_image_stats(project_path=storage_dir, sample_procent=5)

# # save stats to JSON file
# stat_json_path = os.path.join(storage_dir, "per_image_stats.json")
# with io.open(stat_json_path, "w", encoding="utf-8") as file:
#     str_ = json.dumps(stats, indent=4, separators=(",", ": "), ensure_ascii=False)
#     file.write(str(str_))

# # upload stats to Team files
# dst_path = f"/stats/{PROJECT_ID}/per_image_stats.json"
# file_info = api.file.upload(TEAM_ID, stat_json_path, dst_path)
# print(f"Per image stats uploaded to Team files path: {file_info.path}")

# sly.fs.remove_dir(storage_dir)


class StatsPerImage:
    """
    Important fields of modified stats dict:
        "class_names": [],
        "images_count": [],
        "image_counts_filter_by_id": [],
        "objects_count": [],
        "object_counts_filter_by_id": [],
        "avg_nonzero_area": [],
        "avg_nonzero_count": [],
    """

    @property
    def dataset():
        pass


    @staticmethod
    def prepare_data(stats: dict, meta):
        def _col_name(name, color, icon):
            hexcolor = sly.color.rgb2hex(color)
            return '<div><i class="zmdi {}" style="color:{};margin-right:3px"></i> {}</div>'.format(
                icon, hexcolor, name
            )

        def get_col_name_area(name, color):
            return _col_name(name, color, "zmdi-time-interval")

        def get_col_name_count(name, color):
            return _col_name(name, color, "zmdi-equalizer")

        # prepare table columns
        class_names = ["unlabeled"]
        class_colors = [[0, 0, 0]]
        class_indices_colors = [[0, 0, 0]]
        _name_to_index = {}
        table_columns = ["image id", "image", "dataset", "height", "width", "channels", "unlabeled"]

        for idx, obj_class in enumerate(meta.obj_classes):
            class_names.append(obj_class.name)
            class_colors.append(obj_class.color)
            class_index = idx + 1
            class_indices_colors.append([class_index, class_index, class_index])
            _name_to_index[obj_class.name] = class_index
            table_columns.append(get_col_name_area(obj_class.name, obj_class.color))
            table_columns.append(get_col_name_count(obj_class.name, obj_class.color))

        stats["sum_class_area_per_image"] = [0] * len(class_names)
        stats["sum_class_count_per_image"] = [0] * len(class_names)
        stats["count_images_with_class"] = [0] * len(class_names)
        stats["class_names"] = class_names
        stats["class_indices_colors"] = class_indices_colors
        stats["_name_to_index"] = _name_to_index
        stats["columns"] = table_columns
        stats["data"] = []

    @staticmethod
    def update(stats: dict, image_info: sly.ImageInfo, ann_info, meta, *args, **kwargs):
        BG_COLOR = [0, 0, 0]
        # TEAM_ID = sly.env.team_id()
        # WORKSPACE_ID = sly.env.workspace_id()
        load_dotenv(os.path.expanduser("~/supervisely.env"))
        load_dotenv("local.env")
        # api = sly.api.Api()
        # dataset = api.dataset.get_info_by_id(image_info.dataset_id)

        ann_json = ann_info.annotation
        ann = sly.Annotation.from_json(ann_json, meta)
        class_names = stats["class_names"]
        class_indices_colors = stats["class_indices_colors"]
        _name_to_index = stats["_name_to_index"]
        render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
        render_idx_rgb[:] = BG_COLOR
        ann.draw_class_idx_rgb(render_idx_rgb, _name_to_index)
        stat_area = sly.Annotation.stat_area(render_idx_rgb, class_names, class_indices_colors)
        stat_count = ann.stat_class_count(class_names)

        if stat_area["unlabeled"] > 0:
            stat_count["unlabeled"] = 1

        table_row = []
        table_row.append(image_info.id)

        table_row.append(
            # '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(
            #     api.image.url(TEAM_ID, WORKSPACE_ID, dataset.project_id, dataset.id, image_info.id),
            #     image_info.name,
            # )
            image_info.name
        )

        table_row.append(image_info.dataset_id)
        area_unl = stat_area["unlabeled"] if not np.isnan(stat_area["unlabeled"]) else 0
        table_row.extend(
            [
                stat_area["height"],
                stat_area["width"],
                stat_area["channels"],
                round(area_unl, 2),
            ]
        )
        for idx, class_name in enumerate(class_names):
            cur_area = stat_area[class_name] if not np.isnan(stat_area[class_name]) else 0
            cur_count = stat_count[class_name] if not np.isnan(stat_count[class_name]) else 0
            stats["sum_class_area_per_image"][idx] += cur_area
            stats["sum_class_count_per_image"][idx] += cur_count
            stats["count_images_with_class"][idx] += 1 if stat_count[class_name] > 0 else 0
            if class_name == "unlabeled":
                continue
            table_row.append(round(cur_area, 2))
            table_row.append(round(cur_count, 2))

        if len(table_row) != len(stats["columns"]):
            raise RuntimeError("Values for some columns are missed")

        stats["data"].append(table_row)

    @staticmethod
    def aggregate_calculations(stats: dict):
        pass

import json
import os
import random
import re
import shutil
import time
from datetime import datetime
from typing import List, Literal, Optional

import cv2
import requests
import supervisely as sly
import tqdm
from dotenv import load_dotenv
from PIL import Image
from supervisely._utils import camel_to_snake
from supervisely.io.fs import archive_directory, get_file_name, mkdir

import dataset_tools as dtools
from dataset_tools.repo import download
from dataset_tools.repo.sample_project import (
    download_sample_image_project,
    get_sample_image_infos,
)
from dataset_tools.templates import DatasetCategory, License
from dataset_tools.text.generate_summary import list2sentence

DOWNLOAD_ARCHIVE_TEAMFILES_DIR = "/tmp/supervisely/export/export-to-supervisely-format/"

CITATION_TEMPLATE = (
    "If you make use of the {project_name} data, "
    "please cite the following reference:\n\n"
    "``` bibtex \n@dataset{{{project_name},\n"
    "  author={{{authors}}},\n"
    "  title={{{project_name_full}}},\n"
    "  year={{{year}}},\n"
    "  url={{{homepage_url}}}\n}}\n```\n\n"
    "[Source]({homepage_url})"
)

LICENSE_TEMPLATE = "{project_name_full} is under [{license_name}]({license_url}) license.\n\n[Source]({source_url})"
PUBLICLY_AVAILABLE_LICENSE_TEMPLATE = (
    "The {project_name_full} is publicly available.\n\n[Source]({source_url})"
)
UNKNOWN_LICENSE_TEMPLATE = (
    "License is unknown for the {project_name_full} dataset.\n\n[Source]({source_url})"
)

README_TEMPLATE = "# {project_name_full}\n\n{project_name} is a dataset for {cv_tasks}."

DOWNLOAD_SLY_TEMPLATE = (
    "Dataset **{project_name}** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):\n\n [Download]({download_sly_url})\n\n"
    "As an alternative, it can be downloaded with *dataset-tools* package:\n``` bash\npip install --upgrade dataset-tools\n```"
    "\n\n... using following python code:\n``` python\nimport dataset_tools as dtools\n\ndtools.download(dataset='{project_name}', "
    "dst_dir='~/dataset-ninja/')\n```\n"
    "Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.\n\n"
)

DOWNLOAD_ORIGINAL_TEMPLATE = (
    "Please visit dataset [homepage]({homepage_url}) to download the data. \n\n"
    "Afterward, you have the option to download it in the universal [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format) by utilizing the *dataset-tools* package:\n``` "
    "bash\npip install --upgrade dataset-tools\n```"
    "\n\n... using following python code:\n``` python\nimport dataset_tools as dtools\n\n"
    "dtools.download(dataset='{project_name}', dst_dir='~/dataset-ninja/')\n```\n"
    "Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.\n"
)

DOWNLOAD_NONREDISTRIBUTABLE_TEMPLATE = (
    "Please visit dataset [homepage]({homepage_url}) to download the data. \n"
)


class ProjectRepo:
    def __init__(self, api: sly.Api, project_id: int, settings: dict):
        self.project_id = project_id
        self.project_info = api.project.get_info_by_id(project_id)
        self.project_meta = sly.ProjectMeta.from_json(
            api.project.get_meta(project_id, with_settings=True)
        )
        self.project_stats = api.project.get_stats(self.project_id)
        self.datasets = api.dataset.get_list(project_id)

        self.api = api
        self.team_id = sly.env.team_id()
        self.workspace_id = sly.env.workspace_id()

        def _check_ascii_chars(value):
            try:
                value.encode("ascii")
                return True
            except UnicodeEncodeError:
                return False

        def _has_cyrillic(text: str):
            cyrillic_ranges = [
                (0x0400, 0x04FF),  # Cyrillic
                (0x0500, 0x052F),  # Cyrillic Supplement
                (0x2DE0, 0x2DFF),  # Cyrillic Extended-A
                (0xA640, 0xA69F),  # Cyrillic Extended-B
                (0x1C80, 0x1C8F),  # Cyrillic Extended-C
                # Add more ranges if needed
            ]

            for char in str(text):
                if any(start <= ord(char) <= end for start, end in cyrillic_ranges):
                    return True

            return False

        def _has_actual_preview(preview_id: int) -> bool:
            for dataset in self.datasets:
                for image in api.image.get_list(dataset.id):
                    if preview_id == image.id:
                        return True
            return False

        for key, value in settings.items():
            str_val = str(value)
            if _has_cyrillic(value):
                raise TypeError(f"Cyrillic characters contained in the value of key '{key}'.")
            if "<class" in str_val:
                raise TypeError(f"The settings.py file contains non-instances objects.")

        preview_id = settings.get("preview_image_id")
        if preview_id is None:
            raise ValueError("Please fill the following field in main.py: 'PREVIEW_ID'")
        if not _has_actual_preview(preview_id):
            raise ValueError(f"The provided image with ID={preview_id} ")

        self.__dict__.update(settings)

        self.hide_dataset = self.__dict__.get("hide_dataset", True)
        self.buttons = self.__dict__.get("buttons", None)
        self.explore_datasets = self.__dict__.get("explore_datasets", None)
        self.tags = self.__dict__.get("tags", [])
        self.sly_tag_split = self.__dict__.get("slytagsplit", {})
        self.blog = self.__dict__.get("blog", None)
        self.repository = self.__dict__.get("repository", None)
        self.authors_contacts = self.__dict__.get("authors_contacts", None)
        self.classification_task_classes = None

        if self.class2color is not None:
            self._update_colors()

        self.categories = [self.category.text]
        if self.category.featured:
            self.categories.append("featured")
        if self.category.benchmark:
            self.categories.append("benchmark")
        if self.category.extra is not None:
            if isinstance(self.category.extra, list):
                [self.categories.append(elem.text) for elem in self.category.extra]
            elif isinstance(self.category.extra, DatasetCategory):
                self.categories.append(self.category.extra.text)

        self.download_archive_size = int(self.project_info.size)

        self.limited = (
            {"view_count": 12, "download": False} if not self.license.redistributable else None
        )

        def add_buttons(data, text, icon):
            if data is not None:
                if isinstance(data, str):
                    self.buttons.append({"text": text, "icon": icon, "href": data})
                elif isinstance(data, list):
                    if len(data) > 1:
                        for idx, elem in enumerate(data, start=1):
                            self.buttons.append(
                                {"text": f"{text} {idx}", "icon": icon, "href": elem}
                            )
                        self.buttons[0]["text"] = f"{text} 1 (main)"
                    else:
                        self.buttons.append({"text": text, "icon": icon, "href": data[0]})

        self.buttons = []
        publications = [self.paper, self.blog, self.repository]
        for idx, pub, tit, ico in zip(
            [0, 1, 2],
            publications,
            ["Research Paper", "Blog Post", "Repository"],
            ["pdf", "blog", "code"],
        ):
            if isinstance(pub, (str, list)):
                add_buttons(pub, tit, ico)
            if isinstance(pub, str):
                publications[idx] = [pub]
            elif isinstance(pub, dict):
                for k, v in pub.items():
                    self.buttons.append({"text": k, "icon": ico, "href": v})
                publications[idx] = [*pub.values()]
        self.paper, self.blog, self.repository = publications

        self.images_size = {}  # need to generate images first, then update
        self.download_sly_sample_url = None
        self.download_sample_archive_size = None

        if self.license.source_url is None:
            self.license.source_url = self.homepage_url
        self.original_license_path = "LICENSE.md"
        self.original_citation_path = "CITATION.md"

        self._process_download_link(force=settings.get("force_download_sly_url") or False)
        self._update_custom_data()

    def _update_colors(self):
        items = []

        if self.class2color == "predefined":
            sly.logger.info("Custom classes colors are not specified. Using standard predefined...")
            num_classes = len(self.project_stats["images"]["objectClasses"])
            colors = sly.color.get_predefined_colors(num_classes)
            obj_classes = self.project_meta.obj_classes.items()
            for obj_class, color in zip(obj_classes, colors):
                items.append(obj_class.clone(color=color))
        else:
            sly.logger.info("Custom classes colors are specified. Updating...")

            for obj_class in self.project_meta.obj_classes.items():
                if obj_class.name in self.class2color:
                    items.append(obj_class.clone(color=self.class2color[obj_class.name]))
                else:
                    items.append(obj_class)

        project_meta = sly.ProjectMeta(
            obj_classes=items,
            tag_metas=self.project_meta.tag_metas,
            project_type=self.project_meta.project_type,
        )
        self.api.project.update_meta(self.project_id, project_meta)
        self.project_meta = project_meta

        sly.logger.info("Classes colors are updated.")

    def _process_download_link(self, force: bool = False):
        if not self.license.redistributable:
            self.download_sly_url = None
            sly.logger.info("Dataset is non-redistributable. Skipping creation of download url...")
            return
        tf_urls_path = "/cache/released_datasets.json"

        license_path = "LICENSE.md"
        readme_path = "README.md"
        if sly.fs.file_exists(license_path):
            with open(license_path, "r") as f:
                curr_license_content = f.read()
        # elif not sly.fs.file_exists(license_path) and isinstance(self.license, License.Custom):
        #     raise RuntimeError(
        #         "Aborting creation of download url. Please complete the filling of Custom license first."
        #     )

        force_texts = self.__dict__.get("force_texts") or []

        if not force:
            if self.hide_dataset:
                sly.logger.warn(
                    "Dataset is hidden. To generate download link, unhide dataset with 'HIDE_DATASET=False'"
                )
                self.download_sly_url = "Set 'HIDE_DATASET=False' to generate download link"
                return
            if "https://www.dropbox.com/" in self.project_info.custom_data.get(
                "download_sly_url", ""
            ):
                sly.logger.warn(
                    "Download archive is already stored in the dropbox repositiry. Skipping the creation of download link."
                )
                self.download_sly_url = self.project_info.custom_data["download_sly_url"]
                return
            sly.logger.warn(
                "This is a release version of a dataset. Don't forget to double-check annotations shapes, colors, tags, etc."
            )

        else:
            sly.logger.info("Download sly url is passed with force: 'force_download_sly_url==True'")

        _markdown = {
            "LICENSE": (
                self._build_license(license_path)
                if "license" in force_texts or not sly.fs.file_exists(license_path)
                else curr_license_content
            ),
            "README": self._build_readme(readme_path),
        }

        download.update_sly_url_dict(
            self.api,
            {
                self.project_name: {
                    "id": self.project_info.id,
                    "download_sly_url": self.project_info.custom_data.get("download_sly_url"),
                    "download_original_url": self.project_info.custom_data.get(
                        "download_original_url"
                    ),
                    "markdown": _markdown,
                }
            },
            tf_urls_path,
        )

        files = self.api.file.list(
            self.team_id, DOWNLOAD_ARCHIVE_TEAMFILES_DIR, return_type="fileinfo"
        )

        self.download_sly_url = download.prepare_link(
            self.api, self.api.project.get_info_by_id(self.project_id), force, tf_urls_path, files
        )

        if not force and "https://www.dropbox.com" in self.download_sly_url:
            sly.logger.warn(
                f"Be careful: the '{self.project_info.name}' .tar archive is stored on dropbox repository"
            )
            with requests.get(self.download_sly_url, stream=True) as r:
                if r.status_code == 200:
                    self.download_archive_size = int(r.headers.get("Content-Length"))

            return

        if force and "https://www.dropbox.com" in self.download_sly_url:
            sly.logger.info(
                f"Creating forced download link. Dropbox link will be rewritten automatically."
            )
            return

        def sorting_key(filename):
            match = re.search(r"(\d+)_", filename)
            if match:
                return int(match.group(1))
            else:
                return 0

        files = self.api.file.list(
            self.team_id, DOWNLOAD_ARCHIVE_TEAMFILES_DIR, return_type="fileinfo"
        )

        filenames = [file.name for file in files if self.project_name in file.name]
        if len(filenames) == 0:
            raise FileNotFoundError(
                "There is no download archive generated. Please force the creation of the download url."
            )

        sorted_filenames_desc = sorted(filenames, key=sorting_key, reverse=True)

        teamfiles_archive_path = os.path.join(
            DOWNLOAD_ARCHIVE_TEAMFILES_DIR, sorted_filenames_desc[0]
        )
        file_info = self.api.file.get_info_by_path(self.team_id, teamfiles_archive_path)

        # self.download_sly_url = file_info.full_storage_url if file_info is not None else None
        self.download_archive_size = file_info.sizeb if file_info is not None else -1

        download.update_sly_url_dict(
            self.api,
            {
                self.project_name: {
                    "id": self.project_id,
                    "download_sly_url": self.download_sly_url,
                    "download_original_url": self.download_original_url,
                    "markdown": _markdown,
                }
            },
            tf_urls_path,
        )

    def _update_custom_data(self):
        sly.logger.info("Updating project custom data...")

        custom_data = {
            #####################
            # ! required fields #
            #####################
            "name": self.project_name,
            "fullname": self.project_name_full,
            "cv_tasks": self.cv_tasks,
            "annotation_types": self.annotation_types,
            "applications": [vars(application) for application in self.applications],
            "categories": self.categories,
            "release_year": self.release_year,
            "homepage_url": self.homepage_url,
            "license": self.license.name,
            "license_url": self.license.url,
            "preview_image_id": self.preview_image_id,
            "github_url": self.github_url,
            "github": self.github_url[self.github_url.index("dataset-ninja") :],
            "download_sly_url": self.download_sly_url,
            "download_archive_size": self.download_archive_size,
            "is_original_dataset": self.category.is_original_dataset,
            "sensitive": self.category.sensitive_content,
            "limited": self.limited,
            "buttons": self.buttons,
            "hide_dataset": self.hide_dataset,
            "images_size": self.images_size,
            "download_sly_sample_url": self.download_sly_sample_url,
            "download_sample_archive_size": self.download_sample_archive_size,
            #####################
            # ? optional fields #
            #####################
            "release_date": self.release_date,
            "download_original_url": self.download_original_url,
            "paper": self.paper,
            "blog": self.blog,
            "citation_url": self.citation_url,
            "authors": self.authors,
            "authors_contacts": self.authors_contacts,
            "organization_name": self.organization_name,
            "organization_url": self.organization_url,
            "slytagsplit": self.slytagsplit,
            "classification_task_classes": self.classification_task_classes,
            "tags": self.tags,
            "explore_datasets": self.explore_datasets,
        }

        self.api.project.update_custom_data(self.project_id, custom_data)
        self.project_info = self.api.project.get_info_by_id(self.project_id)
        self.custom_data = self.project_info.custom_data

        sly.logger.info("Successfully updated project custom data.")

    def build_stats(
        self,
        force: Optional[
            List[
                Literal[
                    "all",
                    "ClassBalance",
                    "ClassCooccurrence",
                    "ClassesPerImage",
                    "ObjectsDistribution",
                    "ObjectSizes",
                    "ClassSizes",
                    "ClassesHeatmaps",
                    "ClassesPreview",
                    "ClassTreemap",
                ]
            ]
        ] = None,
        settings: dict = {},
    ):
        sly.logger.info("Starting to build stats...")

        literal_stats = [
            "ClassBalance",
            "ClassCooccurrence",
            "ClassesPerImage",
            "ObjectsDistribution",
            "ObjectSizes",
            "ClassSizes",
            "ClassesHeatmaps",
            "ClassesPreview",
            "ClassesTreemap",
        ]

        if force is None:
            force = []
        elif "all" in force:
            force = literal_stats

        sly.logger.info(f"Following stats are passed with force: {force}")

        cls_prevs_settings = settings.get("ClassesPreview", {})
        heatmaps_settings = settings.get("ClassesHeatmaps", {})
        # previews_settings = settings.get("Previews", {})
        cls_prevs_tags = cls_prevs_settings.get("tags", [])
        # classes_to_tags = settings.get("ClassesToTags", {})

        stat_cache = {}
        stats = [
            dtools.ClassBalance(self.project_meta, self.project_stats, stat_cache=stat_cache),
            dtools.ClassCooccurrence(self.project_meta, cls_prevs_tags),
            dtools.ClassToTagCooccurrence(self.project_meta),
            dtools.TagsImagesOneOfDistribution(self.project_meta),
            dtools.TagsImagesCooccurrence(self.project_meta),
            dtools.TagsObjectsCooccurrence(self.project_meta),
            dtools.ClassesPerImage(
                self.project_meta,
                self.project_stats,
                self.datasets,
                cls_prevs_tags,
                self.sly_tag_split,
                stat_cache=stat_cache,
            ),
            dtools.ObjectsDistribution(self.project_meta),
            dtools.ObjectSizes(self.project_meta, self.project_stats),
            dtools.ClassSizes(self.project_meta),
            dtools.ClassesTreemap(self.project_meta),
        ]

        # if len(classes_to_tags) > 0:
        #     stats.append(
        #         dtools.ClassToTagValCooccurrence(self.project_meta, classes_to_tags),
        #     )

        heatmaps = dtools.ClassesHeatmaps(self.project_meta, self.project_stats)

        if cls_prevs_settings.get("tags") is not None:
            self.classification_task_classes = cls_prevs_settings.pop("tags")

        classes_previews = dtools.ClassesPreview(
            self.project_meta, self.project_info, **cls_prevs_settings
        )
        cls_prevs_settings["tags"] = self.classification_task_classes
        classes_previews_tags = dtools.ClassesPreviewTags(
            self.project_meta, self.project_info, **cls_prevs_settings
        )

        for stat in stats:
            if (
                not sly.fs.file_exists(f"./stats/{stat.basename_stem}.json")
                or stat.__class__.__name__ in force
            ):
                stat.force = True
            if (
                isinstance(stat, dtools.ClassCooccurrence)
                and len(self.project_meta.obj_classes.items() + cls_prevs_tags) == 1
            ):
                stat.force = False
        stats = [stat for stat in stats if stat.force]

        vstats = [heatmaps, classes_previews, classes_previews_tags]

        for vstat in vstats:
            if vstat.__class__.__name__ in force:
                vstat.force = True

        if (
            not sly.fs.file_exists(f"./stats/{heatmaps.basename_stem}.png")
            or heatmaps.__class__.__name__ in force
        ):
            heatmaps.force = True
        if (
            not sly.fs.file_exists(f"./visualizations/{classes_previews.basename_stem}.webm")
            or classes_previews.__class__.__name__ in force
        ):
            if self.classification_task_classes is None:
                classes_previews.force = True
            else:
                classes_previews_tags.force = True

        vstats = [stat for stat in vstats if stat.force]

        srate = 1
        if settings.get("Other") is not None:
            srate = settings["Other"].get("sample_rate", 1)

        if self.project_stats["images"]["total"]["imagesMarked"] == 0:
            sly.logger.info(
                "This is a classification-only dataset. It has zero annotations. Building only ClassesPreview and Poster."
            )
            if classes_previews_tags.force is not True:
                return
            stats = []
            vstats = [vstat for vstat in vstats if isinstance(vstat, dtools.ClassesPreviewTags)]
            heatmaps.force, classes_previews.force, classes_previews_tags.force = False, False, True

        dtools.count_stats(
            self.api, self.project_id, self.project_stats, stats=stats + vstats, sample_rate=srate
        )

        sly.logger.info("Saving stats...")
        for stat in stats:
            sly.logger.info(f"Saving {stat.basename_stem}...")
            result_json = stat.to_json()
            if type(result_json) is list and len(result_json) > 0:
                mkdir(f"./stats/one_of_string_tags/")
                for curr_result_json in result_json:
                    tag_name = curr_result_json["data"][0][0].replace("/", " ")
                    with open(f"./stats/one_of_string_tags/{tag_name}.json", "w") as f:
                        json.dump(curr_result_json, f)

                result_json = None

            if result_json is not None:
                with open(f"./stats/{stat.basename_stem}.json", "w") as f:
                    json.dump(result_json, f)
            try:
                stat.to_image(f"./stats/{stat.basename_stem}.png")
            except TypeError:
                pass

        if len(vstats) > 0:
            if heatmaps.force:
                heatmaps.to_image(f"./stats/{heatmaps.basename_stem}.png", **heatmaps_settings)
            if classes_previews.force:
                classes_previews.animate(f"./visualizations/{classes_previews.basename_stem}.webm")
            elif classes_previews_tags.force:  # classification-only dataset
                classes_previews_tags.animate(
                    f"./visualizations/{classes_previews.basename_stem}.webm"
                )

        sly.logger.info("Successfully built and saved stats.")

    def build_visualizations(
        self,
        force: Optional[
            List[Literal["all", "Poster", "SideAnnotationsGrid", "HorizontalGrid", "VerticalGrid"]]
        ] = None,
        settings: dict = {},
    ):
        sly.logger.info("Starting to build visualizations...")

        if force is None:
            force = []
        elif "all" in force:
            force = ["Poster", "SideAnnotationsGrid", "HorizontalGrid", "VerticalGrid"]

        sly.logger.info(f"Following visualizations are passed with force: {force}")

        poster_settings = settings.get("Poster", {})
        side_annots_settings = settings.get("SideAnnotationsGrid", {})
        hor_grid_settings = settings.get("HorizontalGrid", {})
        vert_grid_settings = settings.get("VerticalGrid", {})

        renderers = [
            dtools.Poster(self.project_id, self.project_meta, **poster_settings),
            dtools.SideAnnotationsGrid(self.project_id, self.project_meta, **side_annots_settings),
        ]
        animators = [
            dtools.HorizontalGrid(self.project_id, self.project_meta, **hor_grid_settings),
            dtools.VerticalGrid(self.project_id, self.project_meta, **vert_grid_settings),
        ]

        for vis in renderers + animators:
            if (
                not sly.fs.file_exists(f"./visualizations/{vis.basename_stem}.png")
                or vis.__class__.__name__ in force
            ):
                vis.force = True
        renderers, animators = [r for r in renderers if r.force], [a for a in animators if a.force]

        for a in animators:
            if (
                not sly.fs.file_exists(f"./visualizations/{a.basename_stem}.webm")
                or a.__class__.__name__ in force
            ):
                a.force = True
        animators = [a for a in animators if a.force]

        if self.project_stats["images"]["total"]["imagesMarked"] == 0:
            sly.logger.info(
                "This is a classification-only dataset. It has zero annotations. Building only the Poster."
            )
            renderers, animators = [
                r for r in renderers if r.__class__.__name__ == "Poster" and r.force
            ], []

        # ? Download fonts from: https://fonts.google.com/specimen/Fira+Sans
        dtools.prepare_renders(
            self.project_id,
            renderers=renderers + animators,
            sample_cnt=30,
        )

        sly.logger.info("Saving visualizations...")

        for vis in renderers + animators:
            vis.to_image(f"./visualizations/{vis.basename_stem}.png")
        for a in animators:
            a.animate(f"./visualizations/{a.basename_stem}.webm")

        sly.logger.info("Successfully built and saved visualizations.")

        img = cv2.imread("./stats/classes_heatmaps.png")
        if img is not None:
            self.images_size["classes_heatmaps.png"] = [img.shape[1], img.shape[0]]

        for filename in os.listdir("./visualizations/"):
            if filename.lower().endswith(".png"):
                image_path = os.path.join("./visualizations/", filename)
                img = cv2.imread(image_path)
                if img is not None:
                    height, width, _ = img.shape
                    self.images_size[filename] = [width, height]
            elif filename.lower().endswith((".mp4", ".webm")):
                video_path = os.path.join("./visualizations/", filename)
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.images_size[filename] = [width, height]
                    cap.release()

    def build_demo(self, force: bool = False):
        if not self.license.redistributable:
            sly.logger.info(
                "Dataset is non-redistributable. Skipping creation of demo sample project..."
            )
            return

        storage_dir = sly.app.get_data_dir()
        # workspace_id = sly.env.workspace_id()
        workspace_id_sample_projects = 118

        sample_project_name = f"{self.project_info.name} demo"
        sample_project_exists = self.api.project.exists(
            workspace_id_sample_projects, sample_project_name
        )
        sample_project_info = self.api.project.get_info_by_name(
            workspace_id_sample_projects, sample_project_name
        )

        buffer_project_dir = os.path.join(storage_dir, sample_project_name)
        archive_name = self.project_info.name.lower().replace(" ", "-") + ".tar"
        buffer_project_dir_archive = os.path.join(storage_dir, archive_name)
        teamfiles_archive_path = f"/sample-projects/{archive_name}"

        if not force:
            if sample_project_exists or self.hide_dataset:
                hide_msg = " is hidden with 'HIDE_DATASET=True'" if self.hide_dataset else None
                exists_msg = " already exists" if sample_project_exists else None
                msg_ = [item for item in [hide_msg, exists_msg] if item is not None]
                msg = f"Skipping building of demo project: '{sample_project_name}'{', and'.join(msg_)}."
                sly.logger.info(msg)

                self._demo_update_archivesize(teamfiles_archive_path)
                return

        else:
            sly.logger.info("Demo sample project is passed with force: 'force_demo==True'")

        sly.logger.info("Start to build demo sample project...")

        if self.project_stats["images"]["total"]["imagesMarked"] == 0:
            class_balance_json = None
        else:
            with open("./stats/class_balance.json", "r") as f:
                class_balance_json = json.load(f)

        is_classification_cvtask = True

        is_classification_cvtask = True if self.classification_task_classes is not None else False
        img_infos_sample = get_sample_image_infos(
            self.api,
            self.project_info,
            self.project_stats,
            class_balance_json,
            is_classification_cvtask,
        )

        if img_infos_sample is None:
            sly.logger.info("Dataset is small. Skipping building of demo.")
        elif len(img_infos_sample) == 0:
            raise ValueError(
                "Length of sample images set is zero. Please, check that 'class_balance.json' contains actual images reference_id. If not, rebuild the stat."
            )
        else:
            with tqdm.tqdm(
                desc="Download sample project to buffer", total=len(img_infos_sample)
            ) as pbar:
                if os.path.isdir(buffer_project_dir):
                    shutil.rmtree(buffer_project_dir)

                download_sample_image_project(
                    self.api,
                    self.project_id,
                    img_infos_sample,
                    buffer_project_dir,
                    progress_cb=pbar,
                )
                self._build_license(f"{buffer_project_dir}/LICENSE.md", self.original_license_path)
                self._build_readme(f"{buffer_project_dir}/README.md")

            with tqdm.tqdm(
                desc="Upload sample project to instance", total=len(img_infos_sample)
            ) as pbar:
                if sample_project_exists:
                    self.api.project.remove(sample_project_info.id)
                sly.upload_project(
                    buffer_project_dir,
                    self.api,
                    workspace_id_sample_projects,
                    sample_project_name,
                    progress_cb=pbar,
                )

            sly.logger.info("Start making arhive of a sample project")
            archive_directory(buffer_project_dir, buffer_project_dir_archive)

            with tqdm.tqdm(
                desc="Upload archive to Team files",
                total=len(img_infos_sample),
                unit="B",
                unit_scale=True,
            ) as pbar:
                self.api.file.upload(
                    self.team_id,
                    buffer_project_dir_archive,
                    teamfiles_archive_path,
                    progress_cb=pbar,
                )

            sly.logger.info("Archive with sample project was uploaded to teamfiles")

        self._demo_update_archivesize(teamfiles_archive_path)

    def _demo_update_archivesize(self, teamfiles_archive_path):
        file_info = self.api.file.get_info_by_path(self.team_id, teamfiles_archive_path)

        self.download_sly_sample_url = file_info.full_storage_url if file_info is not None else None
        self.download_sample_archive_size = file_info.sizeb if file_info is not None else None

    def build_texts(
        self,
        force: Optional[
            List[Literal["all", "citation", "license", "readme", "download", "summary"]]
        ] = None,
        preview_class: Optional[
            Literal["ClassesPreview", "HorizontalGrid", "SideAnnotationsGrid"]
        ] = "ClassesPreview",
    ):
        self._update_custom_data()
        self._update_repos_list()
        sly.logger.info("Starting to build texts...")

        if force is None:
            force = []
        elif "all" in force:
            force = ["summary", "citation", "license", "readme", "download"]

        sly.logger.info(f"Following texts are passed with force: {force}")

        if preview_class is None:
            preview_class = "ClassesPreview"

        citation_path = self.original_citation_path
        license_path = self.original_license_path
        readme_path = "README.md"
        download_path = "DOWNLOAD.md"
        summary_path = "SUMMARY.md"

        if "citation" in force or not sly.fs.file_exists(citation_path):
            self._build_citation(citation_path)

        if "license" in force or not sly.fs.file_exists(license_path):
            self._build_license(license_path)

        self._build_readme(readme_path)
        self._build_download(download_path)

        # if "summary" in force or not sly.fs.file_exists(summary_path):
        self._build_summary(summary_path, preview_class=preview_class)

    def _build_summary(self, summary_path, preview_class):
        classname2path = {
            "ClassesPreview": "visualizations/classes_preview.webm",
            "HorizontalGrid": "visualizations/horizontal_grid.png",
            "SideAnnotationsGrid": "visualizations/side_annotations_grid.png",
            "Poster": "visualizations/poster.png",
            "HorizontalGridAnimated": "visualizations/horizontal_grid.webm",
            "VerticalGridAnimated": "visualizations/vertical_grid.webm",
        }

        summary_data = dtools.get_summary_data_sly(self.project_info)

        if preview_class in classname2path.keys() and sly.fs.file_exists(
            f"./{classname2path[preview_class]}"
        ):
            vis_url = f"{self.custom_data['github_url']}/raw/main/{classname2path[preview_class]}"
        else:
            vis_url = None

        summary_content = dtools.generate_summary_content(
            summary_data,
            vis_url=vis_url,
        )

        with open(summary_path, "w") as summary_file:
            summary_file.write(summary_content)

    def _build_citation(self, citation_path):
        if self.citation_url is not None:
            if not os.path.exists(citation_path):
                citation_content = (
                    f"If you make use of the {self.project_name} data, "
                    f"please cite the following reference:\n\n"
                    "``` bibtex\nPASTE HERE CUSTOM CITATION FROM THE SOURCE URL\n```\n\n"
                    f"[Source]({self.citation_url})"
                )
                with open(citation_path, "w") as citation_file:
                    citation_file.write(citation_content)
                    sly.logger.warning("You must update 'CITATION.md' manually.")

            sly.logger.warning("'CITATION.md' already exists. Skipping citation building...")
            return

        sly.logger.info("Starting to build citation...")

        citation_content = CITATION_TEMPLATE.format(
            project_name_full=self.project_name_full,
            authors=" and ".join(self.authors or []),
            project_name=self.project_name,
            homepage_url=self.homepage_url,
            year=self.release_year,
        )

        with open(citation_path, "w") as citation_file:
            citation_file.write(citation_content)

        sly.logger.info("Successfully built and saved citation.")

    def _build_license(self, license_path: str, original_license_path: str = "") -> str:
        sly.logger.info("Starting to build license...")

        if isinstance(self.license, License.Custom):
            if sly.fs.file_exists(original_license_path) and original_license_path != "":
                with open(original_license_path, "r") as license_file:
                    license_content = license_file.read()
            else:
                license_content = (
                    f"ADD CUSTOM LICENSE MANUALLY\n\n[Source]({self.license.source_url})"
                )
                sly.logger.warning("Custom license must be added manually.")
        elif isinstance(self.license, License.Unknown):
            license_content = UNKNOWN_LICENSE_TEMPLATE.format(
                project_name_full=self.project_name_full,
                source_url=self.license.source_url,
            )
        elif isinstance(self.license, License.PubliclyAvailable):
            license_content = PUBLICLY_AVAILABLE_LICENSE_TEMPLATE.format(
                project_name_full=self.project_name_full,
                source_url=self.license.source_url,
            )
        else:
            license_content = LICENSE_TEMPLATE.format(
                project_name_full=self.project_name_full,
                license_name=self.license.name,
                license_url=self.license.url,
                source_url=self.license.source_url,
            )

        with open(license_path, "w") as license_file:
            license_file.write(license_content)

        sly.logger.info("Successfully built and saved license.")
        return license_content

    def _build_readme(self, readme_path) -> str:
        sly.logger.info("Starting to build readme...")

        readme_content = README_TEMPLATE.format(
            project_name_full=self.project_name_full,
            project_name=self.project_name,
            cv_tasks=list2sentence(self.cv_tasks, "task"),
        )

        with open(readme_path, "w") as readme_file:
            readme_file.write(readme_content)

        sly.logger.info("Successfully built and saved readme.")
        return readme_content

    def _build_download(self, download_path):
        sly.logger.info("Starting to build download...")

        if self.license.redistributable:
            if self.download_original_url is not None:
                download_content = DOWNLOAD_SLY_TEMPLATE.format(
                    project_name=self.project_name,
                    download_sly_url=self.download_sly_url,
                )
                if isinstance(self.download_original_url, str):
                    download_content += f"The data in original format can be [downloaded here]({self.download_original_url})."
                if isinstance(self.download_original_url, dict):
                    download_content += "The data in original format can be downloaded here:\n\n"
                    for key, val in self.download_original_url.items():
                        download_content += f"- [{key}]({val})\n"
            else:
                download_content = DOWNLOAD_ORIGINAL_TEMPLATE.format(
                    homepage_url=self.homepage_url,
                    project_name=self.project_name,
                )
        else:
            download_content = DOWNLOAD_NONREDISTRIBUTABLE_TEMPLATE.format(
                homepage_url=self.homepage_url,
            )

        # TODO Set 'HIDE_DATASET=False' to generate download link

        with open(download_path, "w") as download_file:
            download_file.write(download_content)

        sly.logger.info("Successfully built and saved download.")

    def _update_repos_list(self):
        json_path = sly.app.get_data_dir() + "/repos.json"
        tf_repos_path = "/ninja-updater/_repos_list/repos.json"
        self.api.file.download(self.team_id, tf_repos_path, json_path)
        if not sly.fs.file_exists(json_path):
            repos = {}
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                repos = json.load(f)

        dt = datetime.strptime(self.project_info.created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        month = dt.strftime("%B")
        year = dt.strftime("%Y")
        mmyy = f"{month}-{year}"

        if repos.get(mmyy) is None:
            repos[mmyy] = [self.github_url]
        else:
            try:
                repos[mmyy].index(self.github_url)
            except ValueError:
                repos[mmyy].append(self.github_url)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(repos, f)

        self.api.file.upload(self.team_id, json_path, tf_repos_path)

        sly.logger.info(f"The '{tf_repos_path}' was updated.")

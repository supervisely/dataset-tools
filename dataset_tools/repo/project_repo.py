import json
from typing import List, Literal, Optional

import supervisely as sly

import dataset_tools as dtools
from dataset_tools.repo import download
from dataset_tools.templates import DatasetCategory, License

CITATION_TEMPLATE = (
    "If you make use of the {project_name} data, "
    "please cite the following reference:\n\n"
    "``` bibtex \n@dataset{{{project_name},\n"
    "\tauthor={{{authors}}},\n"
    "\ttitle={{{project_name_full}}},\n"
    "\tyear={{{year}}},\n"
    "\turl={{{homepage_url}}}\n}}\n```\n\n"
    "[Source]({homepage_url})"
)

LICENSE_TEMPLATE = "{project_name_full} is under [{license_name}]({license_url}) license."
UNKNOWN_LICENSE_TEMPLATE = "License is unknown for the {project_name_full} dataset."

README_TEMPLATE = "# {project_name_full}\n\n{project_name} is a dataset for {cv_tasks} tasks."

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
        self.project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        self.datasets = api.dataset.get_list(project_id)

        self.api = api
        self.team_id = sly.env.team_id()

        self.__dict__.update(settings)

        self.hide_dataset = self.__dict__.get("hide_dataset", True)
        self.buttons = self.__dict__.get("buttons", None)
        self.explore_datasets = self.__dict__.get("explore_datasets", None)
        self.tags = self.__dict__.get("tags", [])

        if self.class2color:
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
            {"view_count": 10, "download": False} if not self.license.redistributable else None
        )

        if self.paper is not None:
            self.buttons = []
            blogpost_keywords = ["medium.com/", "learnopencv.com/"]
            if isinstance(self.paper, str):
                is_blog_post = any(str2 in self.paper for str2 in blogpost_keywords)
                text = "Blog Post" if is_blog_post else "Research Paper"
                icon = "paper" if is_blog_post else "pdf"
                self.buttons.append({"text": text, "icon": icon, "href": self.paper})
                if is_blog_post:
                    self.tags.append("has_blogpost_as_source")
            elif isinstance(self.paper, list):
                [
                    self.buttons.append(
                        {"text": f"Research Paper {idx}", "icon": "pdf", "href": elem}
                    )
                    for idx, elem in enumerate(self.paper, start=1)
                ]
                self.buttons[0]["text"] = "Research Paper 1 (main)"

        self._process_download_link()
        self._update_custom_data()

    def _update_colors(self):
        sly.logger.info("Custom classes colors are specified. Updating...")

        items = []
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

        sly.logger.info("Custom classes colors are updated.")

    def _process_download_link(self):
        tf_urls_path_hidden = "/cache/download_urls/hidden_datasets.json"
        tf_urls_path_released = "/cache/download_urls/released_datasets.json"

        tf_urls_path = tf_urls_path_hidden if self.hide_dataset else tf_urls_path_released

        if not self.hide_dataset:
            sly.logger.warn(
                "This is a release version of a dataset. Don't forget to double-check annotations shapes, colors, tags, etc."
            )

        self.download_sly_url = download.prepare_link(
            self.api,
            self.project_info,
            tf_urls_path,
        )
        download.update_sly_url_dict(
            self.api,
            {
                self.project_name: {
                    "id": self.project_id,
                    "download_sly_url": self.download_sly_url,
                    "download_original_url": self.download_original_url,
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
            #####################
            # ? optional fields #
            #####################
            "release_date": self.release_date,
            "download_original_url": self.download_original_url,
            "paper": self.paper,
            "citation_url": self.citation_url,
            "authors": self.authors,
            "organization_name": self.organization_name,
            "organization_url": self.organization_url,
            "slytagsplit": self.slytagsplit,
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
                    "Previews",
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
            "Previews",
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

        stat_cache = {}
        stats = [
            dtools.ClassBalance(self.project_meta, stat_cache=stat_cache),
            dtools.ClassCooccurrence(self.project_meta),
            dtools.ClassesPerImage(self.project_meta, self.datasets, stat_cache=stat_cache),
            dtools.ObjectsDistribution(self.project_meta),
            dtools.ObjectSizes(self.project_meta),
            dtools.ClassSizes(self.project_meta),
            dtools.ClassesTreemap(self.project_meta),
        ]
        heatmaps = dtools.ClassesHeatmaps(self.project_meta)
        classes_previews = dtools.ClassesPreview(
            self.project_meta, self.project_info.name, **cls_prevs_settings
        )
        # previews = dtools.Previews(
        #     self.project_id, self.project_meta, self.api, self.team_id, **previews_settings
        # )

        for stat in stats:
            if (
                not sly.fs.file_exists(f"./stats/{stat.basename_stem}.json")
                or stat.__class__.__name__ in force
            ):
                stat.force = True
            if (
                isinstance(stat, dtools.ClassCooccurrence)
                and len(self.project_meta.obj_classes.items()) == 1
            ):
                stat.force = False
        stats = [stat for stat in stats if stat.force]

        vstats = [heatmaps, classes_previews]  # , previews]

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
            classes_previews.force = True
        # if not self.api.file.dir_exists(self.team_id, f"/dataset/{self.project_id}/renders/"):
        #     previews.force = True
        vstats = [stat for stat in vstats if stat.force]

        srate = 1
        if settings.get("Other") is not None:
            srate = settings["Other"].get("sample_rate", 1)

        dtools.count_stats(self.project_id, stats=stats + vstats, sample_rate=srate)

        sly.logger.info("Saving stats...")
        for stat in stats:
            sly.logger.info(f"Saving {stat.basename_stem}...")
            if stat.to_json() is not None:
                with open(f"./stats/{stat.basename_stem}.json", "w") as f:
                    json.dump(stat.to_json(), f)
            try:
                stat.to_image(f"./stats/{stat.basename_stem}.png")
            except TypeError:
                pass

        if len(vstats) > 0:
            if heatmaps.force:
                heatmaps.to_image(f"./stats/{heatmaps.basename_stem}.png", **heatmaps_settings)
            if classes_previews.force:
                classes_previews.animate(f"./visualizations/{classes_previews.basename_stem}.webm")
            # if previews.force:
            # previews.close()

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

    def build_texts(
        self,
        force: Optional[
            List[Literal["all", "citation", "license", "readme", "download", "summary"]]
        ] = None,
        preview_class: Optional[
            Literal["ClassesPreview", "HorizontalGrid", "SideAnnotationsGrid"]
        ] = "ClassesPreview",
    ):
        sly.logger.info("Starting to build texts...")

        if force is None:
            force = []
        elif "all" in force:
            force = ["summary", "citation", "license", "readme", "download"]

        sly.logger.info(f"Following texts are passed with force: {force}")

        if preview_class is None:
            preview_class = "ClassesPreview"

        citation_path = "CITATION.md"
        license_path = "LICENSE.md"
        readme_path = "README.md"
        download_path = "DOWNLOAD.md"
        summary_path = "SUMMARY.md"

        if "citation" in force or not sly.fs.file_exists(citation_path):
            self._build_citation(citation_path)

        if "license" in force or not sly.fs.file_exists(license_path):
            self._build_license(license_path)

        if "readme" in force or not sly.fs.file_exists(readme_path):
            self._build_readme(readme_path)

        if "download" in force or not sly.fs.file_exists(download_path):
            self._build_download(download_path)

        if "summary" in force or not sly.fs.file_exists(summary_path):
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
        sly.logger.info("Starting to build citation...")

        if self.citation_url is not None:
            citation_content = (
                f"If you make use of the {self.project_name} data, "
                f"please cite the following reference:\n\n"
                "``` bibtex\nPASTE HERE CUSTOM CITATION FROM THE SOURCE URL\n```\n\n"
                f"[Source]({self.citation_url})"
            )
        else:
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
        if self.citation_url is not None:
            sly.logger.warning("You must update CITATION.md manually.")

    def _build_license(self, license_path):
        sly.logger.info("Starting to build license...")

        if isinstance(self.license, License.Custom):
            sly.logger.warning("Custom license must be added manually.")
            return

        if self.license.name == "unknown":
            license_content = UNKNOWN_LICENSE_TEMPLATE.format(
                project_name_full=self.project_name_full,
            )
        else:
            license_content = LICENSE_TEMPLATE.format(
                project_name_full=self.project_name_full,
                license_name=self.license.name,
                license_url=self.license.url,
            )

        with open(license_path, "w") as license_file:
            license_file.write(license_content)

        sly.logger.info("Successfully built and saved license.")

    def _build_readme(self, readme_path):
        sly.logger.info("Starting to build readme...")

        readme_content = README_TEMPLATE.format(
            project_name_full=self.project_name_full,
            project_name=self.project_name,
            cv_tasks=", ".join(self.cv_tasks),
        )

        with open(readme_path, "w") as readme_file:
            readme_file.write(readme_content)

        sly.logger.info("Successfully built and saved readme.")

    def _build_download(self, download_path):
        sly.logger.info("Starting to build download...")

        if self.license.redistributable:
            if self.download_original_url is not None:
                download_content = DOWNLOAD_SLY_TEMPLATE.format(
                    project_name=self.project_name,
                    download_sly_url=self.download_sly_url,
                )
                if isinstance(self.download_original_url, str):
                    download_content += f"The data in original format can be [downloaded here]({self.download_original_url})"
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

        with open(download_path, "w") as download_file:
            download_file.write(download_content)

        sly.logger.info("Successfully built and saved download.")

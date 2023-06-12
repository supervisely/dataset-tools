import json

from typing import List, Optional, Literal

import supervisely as sly

from dataset_tools.repo import download
from dataset_tools.templates import License
import dataset_tools as dtools


CITATION_TEMPLATE = (
    "If you make use of the {project_name_full} data, "
    "please cite the following reference (to be prepared after the challenge workshop) "
    "in any publications:\n\n"
    "```\n@misc{{{project_name},\n"
    '\tauthor = "TO BE FILLED MANUALLY!",\n'
    '\ttitle = "{project_name_full}",\n'
    '\thowpublished = "{homepage_url}"\n}}\n```\n\n'
    "[🔗 Source]({homepage_url})"
)

LICENSE_TEMPLATE = "{project_name_full} is under [{license_name}]({license_url}) license."

README_TEMPLATE = "# {project_name_full}\n\n{project_name} is a dataset for {cv_tasks} tasks."

DOWNLOAD_SLY_TEMPLATE = (
    "Dataset **{project_name}** can be downloaded in Supervisely format:\n\n [Download]({download_sly_url})\n\n"
    "As an alternative, it can be downloaded with dataset-tools package:\n``` bash\npip install --upgrade dataset-tools\n```"
    "\n\n... using following python code:\n``` python\nimport dataset_tools as dtools\n\ndtools.download(dataset='{project_name}', "
    "dst_path='~/dtools/datasets/{project_name}.tar')\n```\n"
)

DOWNLOAD_ORIGINAL_TEMPLATE = (
    "Please visit dataset [homepage]({homepage_url}) to download the data. \n\n"
    "Afterward, you have the option to download it in the universal supervisely format by utilizing the dataset-tools package:\n``` "
    "bash\npip install --upgrade dataset-tools\n```"
    "\n\n... using following python code:\n``` python\nimport dataset_tools as dtools\n\n"
    "dtools.download(dataset='{project_name}', dst_path='~/dtools/datasets/{project_name}.tar')\n```\n"
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

        if self.class2color:
            self._update_colors()

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
        project_meta = sly.ProjectMeta(obj_classes=items)
        self.api.project.update_meta(self.project_id, project_meta)
        self.project_meta = project_meta

        sly.logger.info("Custom classes colors are updated.")

    def _process_download_link(self):
        self.download_sly_url = download.prepare_link(self.api, self.project_info)
        download.update_sly_url_dict(
            {
                self.project_name: {
                    "id": self.project_id,
                    "download_sly_url": self.download_sly_url,
                    "download_original_url": self.download_original_url,
                }
            },
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
            "industries": self.industries,
            "release_year": self.release_year,
            "homepage_url": self.homepage_url,
            "license": self.license.name,
            "license_url": self.license.url,
            "preview_image_id": self.preview_image_id,
            "github_url": self.github_url,
            "github": self.github_url[self.github_url.index("dataset-ninja") :],
            "download_sly_url": self.download_sly_url,
            #####################
            # ? optional fields #
            #####################
            "download_original_url": self.download_original_url,
            "paper": self.paper,
            "citation_url": self.citation_url,
            "organization_name": self.organization_name,
            "organization_url": self.organization_url,
            "tags": self.tags,
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
                ]
            ]
        ] = None,
    ):
        sly.logger.info("Starting to build stats...")

        if force is None:
            force = []
        elif "all" in force:
            force = [
                "ClassBalance",
                "ClassCooccurrence",
                "ClassesPerImage",
                "ObjectsDistribution",
                "ObjectSizes",
                "ClassSizes",
                "ClassesHeatmaps",
                "ClassesPreview",
                "Previews",
            ]

        sly.logger.info(f"Following stats are passed with force: {force}")

        stats = [
            dtools.ClassBalance(self.project_meta),
            dtools.ClassCooccurrence(self.project_meta),
            dtools.ClassesPerImage(self.project_meta, self.datasets),
            dtools.ObjectsDistribution(self.project_meta),
            dtools.ObjectSizes(self.project_meta),
            dtools.ClassSizes(self.project_meta),
        ]
        heatmaps = dtools.ClassesHeatmaps(self.project_meta)
        classes_previews = dtools.ClassesPreview(self.project_meta, self.project_info.name)
        previews = dtools.Previews(self.project_id, self.project_meta, self.api, self.team_id)

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

        vstats = [heatmaps, classes_previews, previews]

        for vstat in vstats:
            if vstat.__class__.__name__ in force:
                vstat.force = True

        if not sly.fs.file_exists(f"./stats/{heatmaps.basename_stem}.png"):
            heatmaps.force = True
        if not sly.fs.file_exists(f"./visualizations/{classes_previews.basename_stem}.webm"):
            classes_previews.force = True
        if not self.api.file.dir_exists(self.team_id, f"/dataset/{self.project_id}/renders/"):
            previews.force = True
        vstats = [stat for stat in [heatmaps, classes_previews, previews] if stat.force]

        dtools.count_stats(
            self.project_id,
            stats=stats + vstats,
            sample_rate=1,
        )

        sly.logger.info("Saving stats...")
        for stat in stats:
            sly.logger.info(f"Saving {stat.basename_stem}...")
            with open(f"./stats/{stat.basename_stem}.json", "w") as f:
                json.dump(stat.to_json(), f)
            stat.to_image(f"./stats/{stat.basename_stem}.png")

        if len(vstats) > 0:
            if heatmaps.force:
                heatmaps.to_image(
                    f"./stats/{heatmaps.basename_stem}.png", draw_style="outside_black"
                )
            if classes_previews.force:
                classes_previews.animate(f"./visualizations/{classes_previews.basename_stem}.webm")
            if previews.force:
                previews.close()

        sly.logger.info("Successfully built and saved stats.")

    def build_visualizations(
        self,
        force: Optional[
            List[Literal["all", "Poster", "SideAnnotationsGrid", "HorizontalGrid", "VerticalGrid"]]
        ] = None,
    ):
        sly.logger.info("Starting to build visualizations...")

        if force is None:
            force = []
        elif "all" in force:
            force = ["Poster", "SideAnnotationsGrid", "HorizontalGrid", "VerticalGrid"]

        sly.logger.info(f"Following visualizations are passed with force: {force}")

        renderers = [
            dtools.Poster(self.project_id, self.project_meta),
            dtools.SideAnnotationsGrid(self.project_id, self.project_meta),
        ]
        animators = [
            dtools.HorizontalGrid(self.project_id, self.project_meta),
            dtools.VerticalGrid(self.project_id, self.project_meta),
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
            sample_cnt=40,
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
                f"PASTE HERE CUSTOM CITATION FROM THE SOURCE URL\n\n[🔗 Source]({self.citation_url})"
            )
        else:
            citation_content = CITATION_TEMPLATE.format(
                project_name_full=self.project_name_full,
                project_name=self.project_name,
                homepage_url=self.homepage_url,
            )

        with open(citation_path, "w") as citation_file:
            citation_file.write(citation_content)

        sly.logger.info("Successfully built and saved citation.")
        sly.logger.warning("You must update CITATION.md manually.")

    def _build_license(self, license_path):
        sly.logger.info("Starting to build license...")

        if isinstance(self.license, License.Custom):
            sly.logger.warning("Custom license must be added manually.")
            return

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

        licensecheck = True
        if self.download_original_url is not None and licensecheck:
            download_content = DOWNLOAD_SLY_TEMPLATE.format(
                project_name=self.project_name,
                download_sly_url=self.download_sly_url,
            )
            if isinstance(self.download_original_url, str):
                download_content += f"The data in original format can be :link: [downloaded here]({self.download_original_url})"
            if isinstance(self.download_original_url, dict):
                download_content += "The data in original format can be downloaded here:\n\n"
                for key, val in self.download_original_url.items():
                    download_content += f"- :link:[{key}]({val})\n"
        elif self.download_original_url is None:
            download_content = DOWNLOAD_ORIGINAL_TEMPLATE.format(
                homepage_url=self.homepage_url,
                project_name=self.project_name,
            )

        with open(download_path, "w") as download_file:
            download_file.write(download_content)

        sly.logger.info("Successfully built and saved download.")

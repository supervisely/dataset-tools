from dataset_tools.image.stats.basestats import BaseStats
from typing import Dict, List, Optional
import supervisely as sly
import numpy as np
from collections import defaultdict
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import PieChart

class OverviewPie(BaseStats):
    """
    This stat calculates and visualizes the distribution of annotated and unlabeled images in the project.
    It creates a pie chart showing the ratio between annotated and unlabeled images.
    """

    CHART_HEIGHT = 250
    BORDER_WIDTH = 0
    def __init__(self, project_meta: sly.ProjectMeta, project_stats: Dict, force: bool = False,
                 stat_cache: dict = None) -> None:
        self._meta = project_meta
        self._project_stats = project_stats
        self.force = force
        self._stat_cache = stat_cache

        self._series = []
        self._refs = defaultdict(list)
        self._type = "pie"
        self._colors = None

        self._update_chart()
        
    def _update_chart(self):
        stats = self._project_stats
        marked_count = stats["images"]["total"]["imagesMarked"]
        not_marked_count = stats["images"]["total"]["imagesNotMarked"]
        self._series = [{"name": "Annotated","data": marked_count},
                              {"name": "Unlabeled","data": not_marked_count}]
        self._refs.setdefault("Annotated", [])
        self._refs.setdefault("Unlabeled", [])

    def clean(self):
        self.__init__(self._meta, self._project_stats)

    def update(self):
        raise NotImplementedError()

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        key = "Annotated" if len(figures) > 0 else "Unlabeled"
        self._refs[key].append(image.id)

    def to_json(self):
        raise NotImplementedError()

    def to_json2(self):
        chart = PieChart("", self._series, self.BORDER_WIDTH, True, self.CHART_HEIGHT, self._type)
        if self._colors is not None:
            chart.set_colors(self._colors)

        chart_json = chart.get_json_data()

        chart_json["options"]["dataLabels"] = {
            "enabled": True,
            "style": {
                "fontSize": "16px",
                "fontWeight": "550",
                "colors": ["black"]
            },
            "dropShadow": {"enabled": False}
        }
        chart_json["options"]["plotOptions"] = {
            "pie": {
			    "customScale": 0.8,
				"dataLabels": {
					"offset": 45
				}
			},
			"donut": {
			    "size": "80%",
				"labels": {
					"show": False
				}
			}
        }

        chart_json["options"]["chart"]["height"] = self.CHART_HEIGHT
        chart_json["referencesCell"] = self._seize_list_to_fixed_size(self._refs, 1000)
        return chart_json
        
    def to_numpy_raw(self):
        return np.array([
            self._series,
            dict(self._refs)
        ], dtype=object)
    
    def sew_chunks(self, chunks_dir: str) -> None:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                self._series = loaded_data[0]
                self._refs = loaded_data[1]


class OverviewDonut(OverviewPie):
    """
    This stat calculates and visualizes the distribution of objects by class in the project.
    It creates a donut chart showing the ratio between different object classes.
    """

    def __init__(self, project_meta: sly.ProjectMeta, project_stats: Dict, force: bool = False,
                 stat_cache: dict = None) -> None:
        super().__init__(project_meta, project_stats, force, stat_cache)
        self._type = "donut"
        self._colors = [item.color for item in self._meta.obj_classes.items()] or None
        self._class_id_to_name = {item.sly_id: item.name for item in self._meta.obj_classes.items()}
        self._update_chart()


    def _update_chart(self):
        self._series = []
        self._refs = defaultdict(list)
        for cls in self._project_stats["images"]["objectClasses"]:
            class_name = cls["objectClass"]["name"]
            self._series.append({"name": class_name, "data": cls["total"]})
            self._refs.setdefault(class_name, [])

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        for figure in figures:
            self._refs[self._class_id_to_name[figure.class_id]].append(image.id)

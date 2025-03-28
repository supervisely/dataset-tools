from dataset_tools.image.stats.basestats import BaseStats
from typing import Dict, List, Optional
import supervisely as sly
import numpy as np
from collections import defaultdict
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo

class Overview(BaseStats):
    CHART_HEIGHT = 200
    def __init__(self, project_meta: sly.ProjectMeta, project_stats: Dict, force: bool = False,
                 stat_cache: dict = None) -> None:
        self._meta = project_meta
        self._project_stats = project_stats
        self.force = force
        self._stat_cache = stat_cache

        self._class_id_to_name = {item.sly_id: item.name for item in self._meta.obj_classes}

        self._refs_ann_pie = defaultdict(list)
        self._ann_pie_json = {
            'height': self.CHART_HEIGHT,
            'options': {'chart': {'type': 'pie'},
                        'dataLabels': {'enabled': True},
                        'labels': [],
                        'stroke': {'width': 3},
                        'title': {'text': ''}},
            'series': [],
            'sly_options': {},
            'type': 'pie'
            }
        self._refs_ann_donut = defaultdict(list)
        self._ann_donut_json = {
            'height': self.CHART_HEIGHT, 
            'options': {'chart': {'type': 'donut'},
                        'colors': [],
                        'dataLabels': {'enabled': True},
                        'labels': [],
                        'stroke': {'width': 3},
                        'title': {'text': ''}},
            'series': [],
            'sly_options': {},
            'type': 'donut'
            }
        # self._tag_pie_json = self._ann_pie_json
        # self._tag_donut_json = self._ann_donut_json

        self._update_charts()
        
    def _update_charts(self):
        stats = self._project_stats
        marked_count = stats["images"]["total"]["imagesMarked"]
        not_marked_count = stats["images"]["total"]["imagesNotMarked"]
        self._ann_pie_json['series'].extend([marked_count, not_marked_count])
        self._ann_pie_json['options']['labels'].extend(["Annotated", "Unlabeled"])
        self._refs_ann_pie.setdefault("Annotated", [])
        self._refs_ann_pie.setdefault("Unlabeled", [])

        for cls in stats["images"]["objectClasses"]:
            self._ann_donut_json['series'].append(cls["total"])
            self._ann_donut_json['options']['labels'].append(cls["objectClass"]["name"])
            self._ann_donut_json['options']['colors'].append(cls["objectClass"]["color"])
            self._refs_ann_donut.setdefault(cls["objectClass"]["name"], [])

        # tagged = stats["imageTags"]["total"]["imagesTagged"]
        # not_tagged = stats["imageTags"]["total"]["imagesNotTagged"]
        # self._tag_pie_json['series'].append(tagged, not_tagged)
        # self._tag_pie_json['labels'].append("Tagged", "Not Tagged")

        # for tag in stats["imageTags"]["items"]:
        #     self._tag_donut_json['series'].append(tag["total"])
        #     self._tag_donut_json['labels'].append(tag["tagMeta"]["name"])
        #     self._tag_donut_json['colors'].append(tag["tagMeta"]["color"])

    def clean(self):
        self.__init__(self._meta, self._project_stats)

    def update(self):
        raise NotImplementedError()

    def update2(self, image: ImageInfo, figures: List[FigureInfo]):
        if len(figures) == 0:
            self._refs_ann_pie["Unlabeled"].append(image.id)
        else:
            self._refs_ann_pie["Annotated"].append(image.id)
        for figure in figures:
            self._refs_ann_donut[self._class_id_to_name[figure.class_id]].append(image.id)

    def to_json(self):
        raise NotImplementedError()

    def to_json2(self):
        self._ann_pie_json['referencesCell'] = self._seize_list_to_fixed_size(self._refs_ann_pie, 1000)
        self._ann_donut_json['referencesCell'] = self._seize_list_to_fixed_size(self._refs_ann_donut, 1000)
        return {
            "annPie": self._ann_pie_json,
            "annDonut": self._ann_donut_json,
            # "tagPie": self._tag_pie_json,
            # "tagDonut": self._tag_donut_json,
        }
        
    def to_numpy_raw(self):
        return np.array([
            self._ann_pie_json,
            self._ann_donut_json,
            self._refs_ann_pie,
            self._refs_ann_donut,
            # self._tag_pie_json,
            # self._tag_donut_json,
        ])
    
    def sew_chunks(self, chunks_dir: str) -> None:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                self._ann_pie_json.update(loaded_data[0])
                self._ann_donut_json.update(loaded_data[1])
                self._refs_ann_pie.update(loaded_data[2])
                self._refs_ann_donut.update(loaded_data[3])
                # self._tag_pie_json = loaded_data[2]
                # self._tag_donut_json = loaded_data[3]

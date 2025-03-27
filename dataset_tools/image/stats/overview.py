from dataset_tools.image.stats.basestats import BaseStats
from typing import Dict, List, Optional
import supervisely as sly
import numpy as np

class Overview(BaseStats):
    def __init__(self, project_meta: sly.ProjectMeta, project_stats: Dict, force: bool = False,
                 stat_cache: dict = None) -> None:
        self._meta = project_meta
        self._project_stats = project_stats
        self.force = force
        self._stat_cache = stat_cache

        self._ann_pie_json = {
            'height': 350,
            'options': {'chart': {'type': 'pie'},
                        'dataLabels': {'enabled': True},
                        'labels': [],
                        'stroke': {'width': 3},
                        'title': {'text': ''}},
            'series': [],
            'sly_options': {},
            'type': 'pie'
            }
        self._ann_donut_json = {
            'height': 350,
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
        self._tag_pie_json = self._ann_pie_json
        self._tag_donut_json = self._ann_donut_json

        self._update_charts()
        
    def _update_charts(self):
        stats = self._project_stats
        marked_count = stats["images"]["total"]["imagesMarked"]
        not_marked_count = stats["images"]["total"]["imagesNotMarked"]
        self._ann_pie_json['series'].append(marked_count, not_marked_count)
        self._ann_pie_json['labels'].append("Annotated", "Unlabeled")

        for cls in stats["images"]["objectClasses"]:
            self._ann_donut_json['series'].append(cls["total"])
            self._ann_donut_json['labels'].append(cls["objectClass"]["name"])
            self._ann_donut_json['colors'].append(cls["objectClass"]["color"])

        tagged = stats["imageTags"]["total"]["imagesTagged"]
        not_tagged = stats["imageTags"]["total"]["imagesNotTagged"]
        self._tag_pie_json['series'].append(tagged, not_tagged)
        self._tag_pie_json['labels'].append("Tagged", "Not Tagged")

        for tag in stats["imageTags"]["items"]:
            self._tag_donut_json['series'].append(tag["total"])
            self._tag_donut_json['labels'].append(tag["tagMeta"]["name"])
            self._tag_donut_json['colors'].append(tag["tagMeta"]["color"])

    def update(self):
        raise NotImplementedError()

    def update2(self):
        pass

    def to_json(self):
        raise NotImplementedError()

    def to_json2(self):
        return {
            "annPie": self._ann_pie_json,
            "annDonut": self._ann_donut_json,
            "tagPie": self._tag_pie_json,
            "tagDonut": self._tag_donut_json,
        }
        
    def to_numpy_raw(self):
        return np.array([
            self._ann_pie_json,
            self._ann_donut_json,
            self._tag_pie_json,
            self._tag_donut_json,
        ])
    
    def sew_chunks(self, chunks_dir: str) -> None:
        files = sly.fs.list_files(chunks_dir, valid_extensions=[".npy"])
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is not None:
                self._ann_pie_json = loaded_data[0]
                self._ann_donut_json = loaded_data[1]
                self._tag_pie_json = loaded_data[2]
                self._tag_donut_json = loaded_data[3]

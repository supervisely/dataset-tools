from typing import Dict, List
import random
import dataframe_image as dfi
import numpy as np
import pandas as pd
import supervisely as sly
from supervisely._utils import camel_to_snake


class BaseStats:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        raise NotImplementedError()

    def clean(self) -> None:
        raise NotImplementedError()

    def to_json(self) -> Dict:
        raise NotImplementedError()

    def to_pandas(self, version2=False) -> pd.DataFrame:
        if version2:
            json = self.to_json2()
        else:
            json = self.to_json()
        try:
            table = pd.DataFrame(data=json["data"], columns=json["columns"])
        except (TypeError, KeyError):
            table = None
        return table

    def to_image(self, path: str, version2=False) -> None:
        """
        Create an image visualizing the results of statistics from a Pandas DataFrame.
        """
        ptable = self.to_pandas(version2)
        if ptable is not None:
            table = ptable[:100]  # max rows == 100
            table = table.iloc[:, :100]  # select the first 100 columns
            table.dfi.export(path, max_rows=-1, max_cols=-1, table_conversion="matplotlib")

    @property
    def basename_stem(self) -> str:
        """Get name of your class for your file system"""
        return camel_to_snake(self.__class__.__name__)

    def sew_chunks(self) -> str:
        raise NotImplementedError()

    def _get_summated_canvas(self, bitmap_masks_rgb: List[np.ndarray]) -> np.ndarray:
        masks1channel = [mask[:, :, 0] for mask in bitmap_masks_rgb]
        stacked = np.stack(masks1channel, axis=0)
        return np.sum(stacked, axis=0)

    def check_overlap(self, masks: List[np.ndarray]) -> bool:
        """Each mask is a bitmap"""
        canvas = self._get_summated_canvas(masks)
        is_overlap = ~np.isin(np.unique(canvas), [0, 1]).all()
        return is_overlap

    def calc_unlabeled_area_in(self, masks: List[np.ndarray]) -> float:
        """Each mask is a bitmap"""
        canvas = self._get_summated_canvas(masks)
        zeros_count = np.count_nonzero(canvas == 0)
        return (zeros_count / canvas.size) * 100

    def project_mask(self, canvas, mask, origin=(0, 0)):
        mask_height, mask_width = mask.shape
        x, y = origin
        canvas[y : y + mask_height, x : x + mask_width] += mask
        return canvas

    def group_equal_masks(self, masks: List[np.ndarray]) -> List[int]:
        masks_1channel = [mask[:, :, 0] for mask in masks]

        groups = {}
        group_id = 0

        for mask in masks_1channel:
            found_match = False

            for key, value in groups.items():
                if np.array_equal(mask.tobytes(), key):
                    found_match = True
                    groups[mask.tobytes()] = value
                    break

            if not found_match:
                groups[mask.tobytes()] = group_id
                group_id += 1

        return [groups[mask.tobytes()] for mask in masks_1channel]

    def _seize_list_to_fixed_size(self, lst, max_elements, seed_value=42):
        rng = random.Random(seed_value)
        while len(lst) > max_elements:
            # Randomly remove an element until the list has the desired number of elements
            lst.pop(rng.randint(0, len(lst) - 1))
        return lst


class BaseVisual:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        raise NotImplementedError()

    def to_image(
        self,
        path: str,
        draw_style: str,
        grid_spacing: int,
        outer_grid_spacing: int,
    ) -> None:
        raise NotImplementedError()

    @property
    def basename_stem(self) -> str:
        """Get name of your class for your file system"""
        return camel_to_snake(self.__class__.__name__)

from typing import Dict, List

import dataframe_image as dfi
import numpy as np
import pandas as pd
import supervisely as sly
from supervisely._utils import camel_to_snake


class BaseStats:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        pass

    def to_json(self) -> Dict:
        pass

    def to_pandas(self) -> pd.DataFrame:
        json = self.to_json()
        try:
            table = pd.DataFrame(data=json["data"], columns=json["columns"])
        except KeyError:
            table = None
        return table

    def to_image(self, path: str) -> None:
        """
        Create an image visualizing the results of statistics from a Pandas DataFrame.
        """
        if self.to_pandas() is not None:
            table = self.to_pandas()[:100]  # max rows == 100
            table = table.iloc[:, :100]  # select the first 100 columns
            table.dfi.export(path, max_rows=-1, max_cols=-1)

    @property
    def basename_stem(self) -> str:
        """Get name of your class for your file system"""
        return camel_to_snake(self.__class__.__name__)

    def check_overlap(self, masks: List[np.ndarray]) -> bool:
        nonzero_masks = [mask for mask in masks if np.any(mask != 0)]
        num_masks = len(nonzero_masks)

        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                overlap = np.logical_and(nonzero_masks[i], nonzero_masks[j])
                if np.any(overlap):
                    return True

        return False


class BaseVisual:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        pass

    def to_image(
        self,
        path: str,
        draw_style: str,
        grid_spacing: int,
        outer_grid_spacing: int,
    ) -> None:
        pass

    @property
    def basename_stem(self) -> str:
        """Get name of your class for your file system"""
        return camel_to_snake(self.__class__.__name__)

import dataframe_image as dfi
import numpy as np
import pandas as pd

import supervisely as sly

from supervisely._utils import camel_to_snake


class BaseStats:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        pass

    def to_json(self) -> dict:
        pass

    def to_pandas(self) -> pd.DataFrame:
        json = self.to_json()
        table = pd.DataFrame(data=json["data"], columns=json["columns"])
        return table

    def to_image(self, path) -> None:
        table = self.to_pandas()[:100]  # max rows == 100
        table.dfi.export(path, max_rows=-1, max_cols=-1)

    @property
    def json_name(self) -> None:
        return camel_to_snake(self.__class__.__name__)

class BaseVisual:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        pass
    
    def to_image(
            self, 
            path: str,
            draw_style: str ,
            grid_spacing: int ,
            outer_grid_spacing: int,
        ) -> None:
        pass

    @property
    def json_name(self) -> None:
        return camel_to_snake(self.__class__.__name__)
import itertools
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import dataframe_image as dfi
import numpy as np
import pandas as pd

from supervisely._utils import camel_to_snake

import supervisely as sly


class BaseStats:
    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
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

import os
import random
from typing import Union

import numpy as np
from tqdm import tqdm

import supervisely as sly


class VerticalGrid:
    def __init__(
        self,
        project: Union[str, int],
        project_meta: sly.ProjectMeta,
        api: sly.Api = None,
        rows: int = 6,
        cols: int = 3,
    ):
        self.project_meta = project_meta

        self._max_size = 1920
        self._rows = rows
        self._cols = cols
        self._aspect_ratio = 9 / 16

        self._local = False if isinstance(project, int) else True
        self._api = api if api is not None else sly.Api.from_env()

    def update(self, data: tuple):
        pass

    def to_image(self, path: str = None):
        if path is None:
            storage_dir = sly.app.get_data_dir()
            sly.fs.clean_dir(storage_dir)
            path = os.path.join(storage_dir, "separated_images_grid.jpeg")
        sly.image.write(path, self._grid)
        sly.logger.info(f"Result grid saved to: {path}")

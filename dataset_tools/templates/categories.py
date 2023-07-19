from typing import Literal

from supervisely import logger
from supervisely._utils import camel_to_snake


class DatasetCategory:
    def __init__(self, featured: bool = False):
        # self.postfix = self.__class__.__qualname__.split(".")[0].lower()
        self.text = camel_to_snake(self.__class__.__name__).replace("_", " ").capitalize()

        self.featured = featured


class Category:
    class Benchmark(DatasetCategory):
        """PASCAL, Cityscapes, COCO"""

        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Agriculture(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

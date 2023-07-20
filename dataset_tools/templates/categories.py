from typing import Literal

from supervisely import logger
from supervisely._utils import camel_to_snake


class DatasetCategory:
    def __init__(self, featured: bool = False):
        # self.postfix = self.__class__.__qualname__.split(".")[0].lower()
        self.text = camel_to_snake(self.__class__.__name__).replace("_", " ").capitalize()

        self.featured = featured


class Category:
    class Agriculture(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Aerial(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Benchmark(DatasetCategory):
        """PASCAL, Cityscapes, COCO"""

        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Biology(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Construction(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Environmental(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class EnergyAndUtilities(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Food(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Gaming(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Manufacturing(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Medical(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Surveillance(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Sports(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Livestock(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Retail(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Robotics(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class SelfDriving(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

    class Safety(DatasetCategory):
        def __init__(self, featured=DatasetCategory().featured):
            super().__init__(featured)

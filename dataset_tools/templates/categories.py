from typing import List, Literal, Union

from supervisely import logger
from supervisely._utils import camel_to_snake


class Category:
    pass


class DatasetCategory:
    def __init__(
        self,
        benchmark: bool = False,
        featured: bool = False,
        extra: Union[Category, List[Category]] = None,
    ):
        # self.postfix = self.__class__.__qualname__.split(".")[0].lower()
        self.text = camel_to_snake(self.__class__.__name__).replace("_", " ").capitalize()

        self.benchmark = benchmark
        self.featured = featured
        self.extra = extra


class Category:
    class General(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Agriculture(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Aerial(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Biology(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Construction(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Drones(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Entertainment(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Environmental(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class EnergyAndUtilities(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Food(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Gaming(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Manufacturing(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Medical(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Satellite(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Surveillance(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Sports(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Livestock(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Retail(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Robotics(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class SelfDriving(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

    class Safety(DatasetCategory):
        def __init__(
            self,
            benchmark=DatasetCategory().benchmark,
            featured=DatasetCategory().featured,
            extra=DatasetCategory().extra,
        ):
            super().__init__(benchmark, featured, extra)

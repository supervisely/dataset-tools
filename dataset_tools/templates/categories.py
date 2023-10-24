from typing import List, Literal, Union

from supervisely import logger
from supervisely._utils import camel_to_snake


class Category:
    pass


class DatasetCategory:
    def __init__(
        self,
        extra: Union[Category, List[Category]] = None,
        benchmark: bool = False,
        featured: bool = False,
        is_original_dataset: bool = True,
        sensitive_content: bool = False,
    ):
        # self.postfix = self.__class__.__qualname__.split(".")[0].lower()
        self.text = camel_to_snake(self.__class__.__name__).replace("_", "-").lower()

        self.extra = extra
        self.benchmark = benchmark
        self.featured = featured
        self.is_original_dataset = is_original_dataset
        self.sensitive_content = sensitive_content


class Category:
    class General(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Tutorial(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Agriculture(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Aerial(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Biology(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Construction(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Drones(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Entertainment(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Environmental(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class EnergyAndUtilities(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Food(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Gaming(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Manufacturing(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Medical(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Satellite(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Science(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Security(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)            

    class Surveillance(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Sports(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Livestock(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Retail(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Robotics(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class SelfDriving(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

    class Safety(DatasetCategory):
        def __init__(
            self,
            extra: Union[Category, List[Category]] = None,
            benchmark: bool = False,
            featured: bool = False,
            is_original_dataset: bool = True,
            sensitive_content: bool = False,
        ):
            super().__init__(extra, benchmark, featured, is_original_dataset, sensitive_content)

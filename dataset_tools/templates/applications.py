from supervisely import logger
from supervisely._utils import camel_to_snake


class DatasetApplications:
    """
    Pass 'is_used=False' to the class instance
    if application is not realized in practice i.e.
    has only a potential field of application.
    """

    def __init__(self, is_used: bool = True):
        self.postfix = self.__class__.__qualname__.split(".")[0].lower()
        self.text = camel_to_snake(self.__class__.__name__).replace("_", " ")
        self.is_used = is_used


class Domain:
    class General(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Industrial(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class GIS(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "geoinformational systems (GIS)"


class Research:
    class Agricultural(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Biological(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Genetic(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class SurfaceDefectDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)


class Industry:
    class GeneralDomain:
        def __new__(cls):
            logger.warn(
                "'Industry.GeneralDomain()' is deprecated. Please use 'Domain.General()' instead."
            )
            return Domain.General()

    class Industrial:
        def __new__(cls):
            logger.warn(
                "'Industry.Industrial()' is deprecated. Please use 'Domain.Industry()' instead."
            )
            return Domain.Industrial()

    class Agricultural(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Agriculture(DatasetApplications):
        def __new__(cls):
            logger.warn(
                "'Industry.Agriculture()' is deprecated. Please use 'Industry.Agricultural()' instead."
            )
            return Industry.Agricultural()

    class Aviation(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Medical(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Energy(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Manufacturing(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Food(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Livestock(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Tourism(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Eco(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class AirDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class SearchAndRescue(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "Search and Rescue (SAR)"

    class Satellite(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Environmental(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class UrbanPlanning(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Construction(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

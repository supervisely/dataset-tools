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
    class ComputerAidedQualityControl(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class DamageDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class DroneInspection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Educational(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Environmental(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Engineering(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Forestry(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class General(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Geospatial(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class GIS(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "geoinformational systems (GIS)"

    class Industrial(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class SurfaceDefectDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Surveillance(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class SmallWeakObjectsDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class ThermalDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class VehicleDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class OCR(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "optical character recognition (OCR)"

    class IoT(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "internet of things (IoT)"


class Research:
    class Agricultural(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class AnomalyDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Biological(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Biomedical(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Genetic(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Geological(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Geospatial(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Ecological(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Engineering(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Environmental(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Materials(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Medical(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Neurobiological(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Space(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class SurfaceDefectDetection(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class UrbanPlanning(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)


class Industry:
    class Agricultural(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Aviation(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Automotive(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Construction(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    # class Eco(DatasetApplications):
    #     def __init__(self, is_used: bool = DatasetApplications().is_used):
    #         super().__init__(is_used)

    class Energy(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Entertainment(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Environmental(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Food(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Fishery(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            
    class Forestry(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Livestock(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Logistics(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Manufacturing(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Marine(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Medical(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class OilAndGas(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "oil & gas"

    class Retail(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Robotics(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Safety(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    # class Satellite(DatasetApplications):
    #     def __init__(self, is_used: bool = DatasetApplications().is_used):
    #         super().__init__(is_used)

    class SearchAndRescue(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)
            self.text = "search and rescue (SAR)"

    class Security(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Shipping(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class SmartCity(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Sports(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Surveillance(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Textile(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Tourism(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class TrafficMonitoring(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class UrbanPlanning(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Utilities(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class WasteRecycling(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

    class Wood(DatasetApplications):
        def __init__(self, is_used: bool = DatasetApplications().is_used):
            super().__init__(is_used)

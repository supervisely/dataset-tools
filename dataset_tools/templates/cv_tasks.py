class CVTask:
    class SemanticSegmentation:
        def __new__(cls):
            return "semantic segmentation"

    class InstanceSegmentation:
        def __new__(cls):
            return "instance segmentation"

    class ObjectDetection:
        def __new__(cls):
            return "object detection"

    class Classification:
        def __new__(cls):
            return "classification"

    class Identification:
        def __new__(cls):
            return "identification"

    class MonocularDepthEstimation:
        def __new__(cls):
            return "monocular depth estimation"

    class CellWiseSegmentation:
        def __new__(cls):
            return "cell-wise segmentation"

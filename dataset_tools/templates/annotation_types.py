class AnnotationType:
    class InstanceSegmentation:
        def __new__(cls):
            return "instance segmentation"

    class MonocularDepthEstimation:
        def __new__(cls):
            return "monocular depth estimation"

    class ObjectDetection:
        def __new__(cls):
            return "object detection"

    class SemanticSegmentation:
        def __new__(cls):
            return "semantic segmentation"

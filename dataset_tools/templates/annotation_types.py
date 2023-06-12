class AnnotationType:
    class SemanticSegmentation:
        def __new__(cls):
            return "semantic segmentation"

    class ObjectDetection:
        def __new__(cls):
            return "object detection"

    class InstanceSegmentation:
        def __new__(cls):
            return "instance segmentation"

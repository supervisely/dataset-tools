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

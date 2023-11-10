class AnnotationType:
    class InstanceSegmentation:
        def __new__(cls):
            return "instance segmentation"

    class SemanticSegmentation:
        def __new__(cls):
            return "semantic segmentation"
        
    class ObjectDetection:
        def __new__(cls):
            return "object detection"

    class ImageLevel:
        def __new__(cls):
            return "image-level"

    class PixelsGroupLevel:
        def __new__(cls):
            return "pixels-group-level"

    class MonocularDepthEstimation:
        def __new__(cls):
            return "monocular depth estimation"
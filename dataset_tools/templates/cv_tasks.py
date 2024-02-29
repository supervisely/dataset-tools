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

    class Counting:
        def __new__(cls):
            return "counting"

    class Localization:
        def __new__(cls):
            return "localization"
            
    class MonocularDepthEstimation:
        def __new__(cls):
            return "monocular depth estimation"
        
    class StereoDepthEstimation:
        def __new__(cls):
            return "stereo depth estimation"        

    class SemiSupervisedLearning:
        def __new__(cls):
            return "semi-supervised learning"

    class SelfSupervisedLearning:
        def __new__(cls):
            return "self-supervised learning"

    class UnsupervisedLearning:
        def __new__(cls):
            return "unsupervised learning"
        
    class WeaklySupervisedLearning:
        def __new__(cls):
            return "weakly-supervised learning"
            

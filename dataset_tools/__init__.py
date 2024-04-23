import pkg_resources  # isort: skip

try:
    __version__ = pkg_resources.require("dataset-tools")[0].version
except TypeError as e:
    __version__ = "development"

from dataset_tools.image.renders.convert import (
    compress_mp4,
    compress_png,
    from_mp4_to_webm,
)
from dataset_tools.image.renders.horizontal_grid import HorizontalGrid
from dataset_tools.image.renders.poster import Poster
from dataset_tools.image.renders.previews import Previews
from dataset_tools.image.renders.separated_anns_grid import SideAnnotationsGrid
from dataset_tools.image.renders.vertical_grid import VerticalGrid
from dataset_tools.image.renders.wrapper import prepare_renders
from dataset_tools.image.stats.class_balance import ClassBalance
from dataset_tools.image.stats.class_cooccurrence import ClassCooccurrence, ClassToTagCooccurrence
from dataset_tools.image.stats.classes_per_image import ClassesPerImage
from dataset_tools.image.stats.heatmaps_for_classes import ClassesHeatmaps
from dataset_tools.image.stats.object_and_class_sizes import (
    ClassesTreemap,
    ClassSizes,
    ObjectSizes,
)
from dataset_tools.image.stats.objects_distribution import ObjectsDistribution
from dataset_tools.image.stats.preview_for_classes import (
    ClassesPreview,
    ClassesPreviewTags,
)
from dataset_tools.image.stats.tags_cooccurrence import (
    TagsImagesCooccurrence,
    TagsObjectsCooccurrence,
    TagsImagesOneOfDistribution,
    TagsObjectsOneOfDistribution,
)
from dataset_tools.image.stats.wrapper import (  # , initialize
    count_images_stats,
    count_stats,
)
from dataset_tools.repo.download import download, prepare_link, update_sly_url_dict
from dataset_tools.repo.project_repo import ProjectRepo
from dataset_tools.text.generate_summary import (
    generate_summary_content,
    get_summary_data,
    get_summary_data_sly,
)

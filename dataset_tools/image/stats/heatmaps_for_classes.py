import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import supervisely as sly
from skimage.transform import resize
from dataset_tools.image.stats.basestats import BaseVisual


class ClassesHeatmaps(BaseVisual):
    """
    Get heatmaps of visual density of aggregated annotations for every class in the dataset
    """

    def __init__(self, project_meta: sly.ProjectMeta, heatmap_img_size: tuple = None, force=False):

        self.force = force
        self._meta = project_meta
        self.classname_heatmap = {}
        self._ds_image_sizes = []
        self.heatmap_image_paths = []

        if heatmap_img_size:
            self._heatmap_img_size = heatmap_img_size
        else:
            self._heatmap_img_size = (720, 1280)

        for obj_class in self._meta.obj_classes:
            self.classname_heatmap[obj_class.name] = np.zeros(
                self._heatmap_img_size + (3,), dtype=np.float32
            )

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        image_height, image_width = ann.img_size
        self._ds_image_sizes.append((image_height, image_width))
        geometry_types_to_heatmap = ["polygon", "rectangle", "bitmap"]
        ann = ann.resize(self._heatmap_img_size)
        for label in ann.labels:
            temp_canvas = np.zeros(self._heatmap_img_size + (3,), dtype=np.uint8)
            if label.geometry.geometry_name() in geometry_types_to_heatmap:
                label.draw(temp_canvas, color=(1, 1, 1))
                self.classname_heatmap[label.obj_class.name] += temp_canvas

    def to_image(
        self,
        path: str,
        draw_style: str = "inside_white",
        grid_spacing: int = 20,
        outer_grid_spacing: int = 20,
    ) -> None:
        """
        Crates image grid with density heatmaps of all possible classes.

        :param path: path to save output image.
        :type path: str.
        :param draw_style: style in which output heatmaps grid will be represented. Possible values: "inside_white", "outside_black"
        :type draw_style: str, optional
        :param grid_spacing: spaces between images. Defaults to 20.
        :type grid_spacing: int, optional
        :param outer_grid_spacing: frame around the overall image. Defaults to 20.
        :type outer_grid_spacing: int, optional
        """
        self._calculate_output_img_size()

        if draw_style == "inside_white":
            self._create_single_images_text_inside(path)
        if draw_style == "outside_black":
            self._create_single_images_text_outside(path)

        img_height, img_width = cv2.imread(self.heatmap_image_paths[0]).shape[:2]
        num_images = len(self.heatmap_image_paths)
        rows = math.ceil(math.sqrt(num_images))
        cols = math.ceil(num_images / rows)

        result_width = cols * (img_width + grid_spacing) - grid_spacing + 2 * outer_grid_spacing
        result_height = rows * (img_height + grid_spacing) - grid_spacing + 2 * outer_grid_spacing

        result_image = Image.new("RGB", (result_width, result_height), "white")

        for i, img_path in enumerate(self.heatmap_image_paths):
            img = Image.open(img_path)
            row = i // cols
            col = i % cols
            x = outer_grid_spacing + col * (img_width + grid_spacing)
            y = outer_grid_spacing + row * (img_height + grid_spacing)
            result_image.paste(img, (x, y))
            sly.api.file_api.silent_remove(img_path)
            self.heatmap_image_paths = []

        result_image.save(path)

    def _create_single_images_text_outside(self, path):
        for heatmap in self.classname_heatmap:
            resized_image = resize(self.classname_heatmap[heatmap], self._heatmap_img_size)
            image_path = os.path.join(os.path.dirname(path), f"{heatmap}.png")
            plt.imsave(image_path, resized_image[:, :, 0])

            image = cv2.imread(image_path)
            image = self._draw_text_below_image(heatmap, image)
            cv2.imwrite(image_path, image)

            self.heatmap_image_paths.append(image_path)

    def _draw_text_below_image(self, text, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        font_scale = 1.0
        text_color = (0, 0, 0)
        thickness = 3
        text_height_percent = 10
        line_spacing_percent = 5

        image_height = image.shape[0]
        text_height = int(image_height * text_height_percent / 100)

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        while text_size[1] > text_height:
            font_scale -= 0.1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        while text_size[1] < text_height:
            font_scale += 0.1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        characters = list(text)
        lines = []
        current_line = characters[0]

        for char in characters[1:]:
            temp_line = current_line + char
            size = cv2.getTextSize(temp_line, font, font_scale, thickness)[0]

            if size[0] <= image.shape[1]:
                current_line = temp_line
            else:
                lines.append(current_line)
                current_line = char

        lines.append(current_line)

        text_x = 10
        text_y = image_height + text_height + 10

        line_spacing = int(image_height * line_spacing_percent / 100)

        result_height = (
            image_height + text_height + (len(lines) - 1) * (text_height + line_spacing) + 40
        )
        result_image = np.zeros((result_height, image.shape[1], 3), dtype=np.uint8)
        result_image.fill(255)
        result_image[:image_height, :] = image

        for line in lines:
            cv2.putText(
                result_image,
                line,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                line_type,
            )
            text_y += text_height + line_spacing

        return result_image

    def _create_single_images_text_inside(self, path):
        for heatmap in self.classname_heatmap:
            font_scale = self._get_optimal_font_scale(heatmap)
            resized_image = resize(self.classname_heatmap[heatmap], self._heatmap_img_size)
            x_pos_center = int(resized_image.shape[1] * 0.5)
            y_pos_percent = int(resized_image.shape[0] * 0.95)

            image_path = os.path.join(os.path.dirname(path), f"{heatmap}.png")
            plt.imsave(image_path, resized_image[:, :, 0])

            image = cv2.imread(image_path)
            text = f"{heatmap}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)
            thickness = 3
            line_type = cv2.LINE_AA
            text_width = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
            text_position = (x_pos_center - int(text_width / 2), y_pos_percent)
            cv2.putText(
                image, text, text_position, font, font_scale, text_color, thickness, line_type
            )
            cv2.imwrite(image_path, image)

            self.heatmap_image_paths.append(image_path)

    def _get_optimal_font_scale(self, text):
        font_scale = 10
        thickness = 3
        text_height_percent = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        while text_width > self._heatmap_img_size[1]:
            font_scale -= 0.1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]

        desired_text_height = (self._heatmap_img_size[0] * text_height_percent) // 100
        font_scale *= desired_text_height / text_height
        return font_scale

    def _calculate_output_img_size(self):
        sizes = np.array(self._ds_image_sizes)
        if np.all(sizes == sizes[0]):
            self._heatmap_img_size = sizes[0]
        else:
            widths = sizes[:, 1]
            heights = sizes[:, 0]
            median_width = np.median(widths)
            median_height = np.median(heights)
            self._heatmap_img_size = (median_height, median_width)

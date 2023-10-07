import math
import os
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize

import supervisely as sly
from dataset_tools.image.stats.basestats import BaseVisual

CURENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURENT_DIR))


class ClassesHeatmaps(BaseVisual):
    """
    Get heatmaps of visual density of aggregated annotations for every class in the dataset
    """

    def __init__(self, project_meta: sly.ProjectMeta, project_stats:dict, heatmap_img_size: tuple = None, force=False):
        self.force = force
        self._meta = project_meta
        self._project_stats = project_stats
        self.classname_heatmap = {}
        self._ds_image_sizes = []
        self.heatmap_image_paths = []
        self._font = None
        self.max_cols = 6

        if heatmap_img_size:
            self._heatmap_img_size = heatmap_img_size
        else:
            self._heatmap_img_size = (720, 1280)

        # TODO remove later
        self._shortlist_cls = None
        if len(self._project_stats['images']['objectClasses']) > 500:
            totals = [(cls['objectClass']['name'], cls['total'] ) for cls in self._project_stats['images']['objectClasses']]
            self._shortlist_cls = [x[0] for x in totals if x[1] > 50]

        if self._shortlist_cls is None:
            for obj_class in self._meta.obj_classes:
                self.classname_heatmap[obj_class.name] = np.zeros(
                    self._heatmap_img_size + (3,), dtype=np.float32
                )
        else:
            for obj_class_name in self._shortlist_cls:
                self.classname_heatmap[obj_class_name] = np.zeros(
                    self._heatmap_img_size + (3,), dtype=np.float32
                )        

        

    def update(self, image: sly.ImageInfo, ann: sly.Annotation) -> None:
        image_height, image_width = ann.img_size
        self._ds_image_sizes.append((image_height, image_width))
        geometry_types_to_heatmap = [
            sly.Polygon.name(),
            sly.Rectangle.name(),
            sly.Bitmap.name(),
            sly.Point.name(),
        ]

        for label in ann.labels:
            if self._shortlist_cls is not None:
                if label.obj_class.name not in self._shortlist_cls:
                    continue

            temp_canvas = np.zeros(ann.img_size + (3,), dtype=np.uint8)
            if label.geometry.name() in geometry_types_to_heatmap:
                if label.geometry.name() == sly.Point.name():
                    label.draw(temp_canvas, color=(1, 1, 1), thickness=5)
                else:
                    label.draw(temp_canvas, color=(1, 1, 1))
                temp_canvas = cv2.resize(temp_canvas, self._heatmap_img_size[::-1])
                self.classname_heatmap[label.obj_class.name] += temp_canvas

    def to_image(
        self,
        path: str,
        draw_style: str = "inside_white",
        rows: Union[int, None] = None,
        cols: Union[int, None] = None,
        grid_spacing: int = 20,
        outer_grid_spacing: int = 20,
        output_width: int = 1920,
        font: str = os.path.join(PARENT_DIR, "fonts/FiraSans-Bold.ttf"),
    ) -> None:
        """
        Crates image grid with density heatmaps of all possible classes.

        :param path: path to save output image.
        :type path: str.
        :param draw_style: style in which output heatmaps grid will be represented. Possible values: "inside_white", "outside_black"
        :type draw_style: str, optional
        :param rows: number of rows. Defaults to None.
        :type rows: int or None, optional
        :param cols: number of columns. Defaults to None.
        :type cols: int or None, optional
        :param grid_spacing: spaces between images. Defaults to 20.
        :type grid_spacing: int, optional
        :param outer_grid_spacing: frame around the overall image. Defaults to 20.
        :type outer_grid_spacing: int, optional
        :param output_width: width of result image. Defaults to 1920 px.
        :type output_width: int, optional
        :param font: path to font file. Defaults to "fonts/FiraSans-Bold.ttf".
        :type font: str, optional
        """
        self._calculate_output_img_size()
        self._font = ImageFont.truetype(font)
        self.output_width = output_width
        self.cols = cols
        self.rows = rows

        if draw_style == "inside_white":
            self._create_single_images_text_inside(path)
        if draw_style == "outside_black":
            self._create_single_images_text_outside(path)

        img_width, img_height = Image.open(self.heatmap_image_paths[0]).size
        self.num_images = len(self.heatmap_image_paths)

        if self.cols is not None and self.rows is None:
            self._get_rows()
        if self.cols is None and self.rows is not None:
            self._get_cols()
        if self.cols is None and self.rows is None:
            self._get_grid_dimensions()

        if self.cols < self.max_cols:
            self._change_output_width_relative_to_cols()

        result_width = (
            self.cols * (img_width + grid_spacing) - grid_spacing + 2 * outer_grid_spacing
        )
        result_height = (
            self.rows * (img_height + grid_spacing) - grid_spacing + 2 * outer_grid_spacing
        )

        result_image = Image.new("RGB", (result_width, result_height), "white")

        for i, img_path in enumerate(self.heatmap_image_paths):
            img = Image.open(img_path)
            row = i // self.cols
            col = i % self.cols
            x = outer_grid_spacing + col * (img_width + grid_spacing)
            y = outer_grid_spacing + row * (img_height + grid_spacing)
            result_image.paste(img, (x, y))
            sly.api.file_api.silent_remove(img_path)
            self.heatmap_image_paths = []

        width_percent = self.output_width / result_width
        output_height = math.ceil(result_height * width_percent)
        result_image = result_image.resize(
            (self.output_width, output_height), Image.Resampling.LANCZOS
        )

        result_image.save(path)

    def _get_rows(self):
        self.rows = math.ceil(self.num_images / self.cols)

    def _get_cols(self):
        self.cols = math.ceil(self.num_images / self.rows)
        if self.cols > self.max_cols:
            self.cols = self.max_cols
            self._get_rows()
            sly.logger.warn(
                f"Your request readjusted to {self.rows}x{self.cols} dimensions due to max possible columns = 6"
            )

    def _get_grid_dimensions(self, aspect_ratio: Union[float, int] = 1.9):
        self.cols = min(max(int(math.sqrt(self.num_images) * aspect_ratio), 1), self.max_cols)
        self.rows = max((self.num_images - 1) // self.cols + 1, 1)

    def _change_output_width_relative_to_cols(self):
        self.output_width = math.ceil(self.output_width / self.max_cols * self.cols)

    def _create_single_images_text_outside(self, path):
        for heatmap in list(self.classname_heatmap.keys()):
            resized_image = resize(self.classname_heatmap[heatmap], self._heatmap_img_size)
            heatmap_name = heatmap.replace("/", "_")
            image_path = os.path.join(os.path.dirname(path), f"{heatmap_name}.png")
            plt.imsave(image_path, resized_image[:, :, 0])

            image = Image.open(image_path)
            image = self._draw_text_below_image(heatmap, image)
            image.save(image_path)

            self.heatmap_image_paths.append(image_path)
            del self.classname_heatmap[heatmap]

    def _draw_text_below_image(self, text, image):
        text_color = (0, 0, 0)
        text_height_percent = 10
        line_spacing_percent = 20

        image_height = image.size[1]
        text_height = math.ceil(image_height * text_height_percent / 100)

        font = self._font.font_variant(size=text_height)

        characters = list(text)
        lines = []
        current_line = characters[0]

        for char in characters[1:]:
            temp_line = current_line + char
            size = font.getsize(temp_line)

            if size[0] <= image.size[0]:
                current_line = temp_line
            else:
                lines.append(current_line)
                current_line = char

        lines.append(current_line)

        line_spacing = int(text_height * line_spacing_percent / 100)

        result_height = (
            image_height + text_height + (len(lines) - 1) * (text_height + line_spacing) + 40
        )
        result_image = Image.new("RGB", (image.size[0], result_height), color=(255, 255, 255))
        result_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(result_image)

        text_x = 10
        text_y = image_height + 10

        for line in lines:
            draw.text((text_x, text_y), line, font=font, fill=text_color)
            text_y += text_height + line_spacing

        return result_image

    def _create_single_images_text_inside(self, path):
        for heatmap in list(self.classname_heatmap.keys()):
            font_size = self._get_optimal_font_size(heatmap)
            resized_image = resize(self.classname_heatmap[heatmap], self._heatmap_img_size)
            x_pos_center = int(resized_image.shape[1] * 0.5)
            y_pos_percent = int((resized_image.shape[0] - font_size) * 0.95)

            heatmap_name = heatmap.replace("/", "_")
            image_path = os.path.join(os.path.dirname(path), f"{heatmap_name}.png")
            plt.imsave(image_path, resized_image[:, :, 0])

            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            font = self._font.font_variant(size=font_size)
            text = f"{heatmap}"
            text_color = (255, 255, 255)
            text_width, _ = draw.textsize(text, font=font)
            text_position = (x_pos_center - int(text_width / 2), y_pos_percent)
            draw.text(text_position, text, font=font, fill=text_color)
            image.save(image_path)

            self.heatmap_image_paths.append(image_path)
            del self.classname_heatmap[heatmap]

    def _get_optimal_font_size(self, text):
        desired_text_width = math.ceil(self._heatmap_img_size[1] * 0.92)
        text_height_percent = 10
        font_size = 10

        font = self._font.font_variant(size=font_size)
        text_width, _ = font.getsize(text)

        while text_width > desired_text_width:
            font_size -= 1
            font = font.font_variant(size=font_size)
            text_width, _ = font.getsize(text)
            font.font_variant

        desired_font_height = math.ceil((self._heatmap_img_size[0] * text_height_percent) // 100)
        desired_font_size = math.ceil(font_size * desired_text_width / text_width)
        desired_font_size = min(desired_font_size, desired_font_height)
        return desired_font_size

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

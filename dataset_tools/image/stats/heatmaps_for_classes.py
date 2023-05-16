import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import supervisely as sly


class ClassesHeatmaps:
    def __init__(self, project_meta: sly.ProjectMeta, heatmap_img_size: tuple = None):
        self._meta = project_meta
        self.classname_heatmap = {}
        self._ds_image_sizes = []
        self.heatmap_image_paths = []

        if heatmap_img_size:
            self._heatmap_img_size = heatmap_img_size
        else:
            self._heatmap_img_size = (800, 1200)

        for obj_class in self._meta.obj_classes:
            self.classname_heatmap[obj_class.name] = np.zeros(
                self._heatmap_img_size + (3,), dtype=np.float32
            )

    def update(self, image: sly.ImageInfo, ann: sly.Annotation):
        image_height, image_width = ann.img_size
        self._ds_image_sizes.append((image_height, image_width))
        geometry_types_to_heatmap = ["polygon", "rectangle", "bitmap"]
        ann = ann.resize(self._heatmap_img_size)
        for label in ann.labels:
            temp_canvas = np.zeros(self._heatmap_img_size + (3,), dtype=np.uint8)
            if label.geometry.geometry_name() in geometry_types_to_heatmap:
                label.draw(temp_canvas, color=(1, 1, 1))
                self.classname_heatmap[label.obj_class.name] += temp_canvas

    def _create_single_images(self, path):
        for heatmap in self.classname_heatmap:
            x_pos_center = int(self.classname_heatmap[heatmap].shape[1] * 0.5)
            y_pos_percent = int(self.classname_heatmap[heatmap].shape[0] * 0.96)

            image_path = f"{path}/{heatmap}.png"
            plt.imsave(image_path, self.classname_heatmap[heatmap][:, :, 0])

            image = cv2.imread(image_path)
            text = f"{heatmap}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            text_color = (255, 255, 255)
            thickness = 3
            line_type = cv2.LINE_AA
            (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_position = (x_pos_center - int(text_width / 2), y_pos_percent)
            cv2.putText(
                image, text, text_position, font, font_scale, text_color, thickness, line_type
            )
            cv2.imwrite(image_path, image)

            sly.logger.info(f"Heatmap image for class [{heatmap}] created")
            self.heatmap_image_paths.append(image_path)

    def to_image(self, path, grid_spacing=20, outer_grid_spacing=20):
        self._create_single_images(path)
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

        save_path = f"{path}/classes_heatmaps.png"
        result_image.save(save_path)
        sly.logger.info(f"Heatmap image for all classes created at {save_path}")

    def _calculate_output_img_size(self):
        sizes = np.array(self._ds_image_sizes)
        self._heatmap_img_size = (
            np.max(sizes[:, 0]),
            np.max(sizes[:, 1]),
        )
        sly.logger.info(f"Max size of {self._heatmap_img_size} for heatmaps calculated")

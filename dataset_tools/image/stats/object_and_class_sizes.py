from collections import defaultdict

import pandas as pd
import supervisely as sly


class ObjectSizes:
    def __init__(self, project_data):
        self.project_meta = project_data.project_meta
        self.datasets = project_data.datasets

    def update(self):
        self.sizes = []
        for dataset in self.datasets:
            image_names = [image_info.name for image_info in dataset.image_infos]
            for image_name, ann in zip(image_names, dataset.anns):
                image_height, image_width = ann.img_size

                for label in ann.labels:
                    if type(label.geometry) not in [sly.Bitmap, sly.Rectangle, sly.Polygon]:
                        continue

                object_data = {
                    "class_name": label.obj_class.name,
                    "image_name": image_name,
                    "dataset_name": dataset.name,
                    "image_size_hw": f"{image_height}x{image_width}",
                }

                object_data.update(calculate_obj_sizes(label, image_height, image_width))

                object_data = list(object_data.values())

                self.sizes.append(object_data)

    def to_json(self):
        options = {"fixColumns": 1}

        res = {
            "columns": [
                "Class name",
                "Image name",
                "Dataset name",
                "Image size",
                "Height",
                "Height",
                "Width",
                "Width",
                "Area",
                "Area",
            ],
            "columnsOptions": [
                {},
                {},
                {},
                {},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
            ],
            "data": self.sizes,
            "options": options,
        }

        return res

    def to_pandas(self):
        json = self.to_json()
        table = pd.DataFrame(json["data"], columns=json["columns"])
        return table


class ClassSizes:
    def __init__(self, project_data):
        self.project_meta = project_data.project_meta
        self.datasets = project_data.datasets
        self.class_titles = [obj_class.name for obj_class in project_data.project_meta.obj_classes]

    def update(self):
        class_heights_px = defaultdict(list)
        class_heights_pc = defaultdict(list)
        class_widths_px = defaultdict(list)
        class_widths_pc = defaultdict(list)
        class_areas_px = defaultdict(list)
        class_areas_pc = defaultdict(list)
        class_object_counts = defaultdict(int)

        for dataset in self.datasets:
            for ann in dataset.anns:
                image_height, image_width = ann.img_size

                for label in ann.labels:
                    if type(label.geometry) not in [sly.Bitmap, sly.Rectangle, sly.Polygon]:
                        continue

                    class_object_counts[label.obj_class.name] += 1

                    obj_sizes = calculate_obj_sizes(label, image_height, image_width)

                    class_heights_px[label.obj_class.name].append(obj_sizes["height_px"])
                    class_heights_pc[label.obj_class.name].append(obj_sizes["height_pc"])
                    class_widths_px[label.obj_class.name].append(obj_sizes["width_px"])
                    class_widths_pc[label.obj_class.name].append(obj_sizes["width_pc"])
                    class_areas_px[label.obj_class.name].append(obj_sizes["area_px"])
                    class_areas_pc[label.obj_class.name].append(obj_sizes["area_pc"])

        self.sizes = []

        for class_title in self.class_titles:
            object_count = class_object_counts[class_title]

            if object_count < 1:
                continue

            class_data = {
                "class_name": class_title,
                "object_count": object_count,
                "min_height_px": min(class_heights_px[class_title]),
                "min_height_pc": min(class_heights_pc[class_title]),
                "max_height_px": max(class_heights_px[class_title]),
                "max_height_pc": max(class_heights_pc[class_title]),
                "avg_height_px": round(
                    sum(class_heights_px[class_title]) / len(class_heights_px[class_title]),
                    2,
                ),
                "avg_height_pc": round(
                    sum(class_heights_pc[class_title]) / len(class_heights_pc[class_title]),
                    2,
                ),
                "min_width_px": min(class_widths_px[class_title]),
                "min_width_pc": min(class_widths_pc[class_title]),
                "max_width_px": max(class_widths_px[class_title]),
                "max_width_pc": max(class_widths_pc[class_title]),
                "avg_width_px": round(
                    sum(class_widths_px[class_title]) / len(class_widths_px[class_title]),
                    2,
                ),
                "avg_width_pc": round(
                    sum(class_widths_pc[class_title]) / len(class_widths_pc[class_title]),
                    2,
                ),
                "min_area_px": min(class_areas_px[class_title]),
                "min_area_pc": min(class_areas_pc[class_title]),
                "max_area_px": max(class_areas_px[class_title]),
                "max_area_pc": max(class_areas_pc[class_title]),
                "avg_area_px": round(
                    sum(class_areas_px[class_title]) / len(class_areas_px[class_title]), 2
                ),
                "avg_area_pc": round(
                    sum(class_areas_pc[class_title]) / len(class_areas_pc[class_title]), 2
                ),
            }

            class_data = list(class_data.values())

            self.sizes.append(class_data)

    def to_json(self):
        options = {"fixColumns": 1}

        res = {
            "columns": [
                "Class name",
                "Object count",
                "Min height",
                "Min height",
                "Max height",
                "Max height",
                "Avg height",
                "Avg height",
                "Min width",
                "Min width",
                "Max width",
                "Max width",
                "Avg width",
                "Avg width",
                "Min area",
                "Min area",
                "Max area",
                "Max area",
                "Avg area",
                "Avg area",
            ],
            "columnsOptions": [
                {},
                {},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
                {"postfix": "px"},
                {"postfix": "%"},
            ],
            "data": self.sizes,
            "options": options,
        }

        return res

    def to_pandas(self):
        json = self.to_json()
        table = pd.DataFrame(json["data"], columns=json["columns"])
        return table


def calculate_obj_sizes(label, image_height, image_width):
    image_area = image_height * image_width

    rect_geometry = label.geometry.to_bbox()

    height_px = rect_geometry.height
    height_pc = round(height_px * 100.0 / image_height, 2)

    width_px = rect_geometry.width
    width_pc = round(width_px * 100.0 / image_width, 2)

    area_px = label.geometry.area
    area_pc = round(area_px * 100.0 / image_area, 2)

    return {
        "height_px": height_px,
        "height_pc": height_pc,
        "width_px": width_px,
        "width_pc": width_pc,
        "area_px": area_px,
        "area_pc": area_pc,
    }

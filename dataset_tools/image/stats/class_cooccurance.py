class ImgClassesCooccurence:
    """
    Important fields of modified stats dict:
        "class_names": [],
        "counters": [],
        "pd_data": [],
    """

    @staticmethod
    def prepare_data(stats: Dict, meta):
        class_names = [cls.name for cls in meta.obj_classes]
        counters = defaultdict(list)
        stats["class_names"] = class_names
        stats["counters"] = counters

    @staticmethod
    def update(stats: Dict, image_info, ann_info, meta, current_dataset):
        ann_json = ann_info.annotation
        ann = sly.Annotation.from_json(ann_json, meta)

        classes_on_image = set()
        for label in ann.labels:
            classes_on_image.add(label.obj_class.name)

        all_pairs = set(
            frozenset(pair) for pair in itertools.product(classes_on_image, classes_on_image)
        )
        for p in all_pairs:
            stats["counters"][p].append((image_info, current_dataset))

    @staticmethod
    def aggregate_calculations(stats: Dict):
        pd_data = []
        class_names = stats["class_names"]
        columns = ["name", *class_names]
        for cls_name1 in class_names:
            cur_row = [cls_name1]
            for cls_name2 in class_names:
                key = str(frozenset([cls_name1, cls_name2]))
                imgs_cnt = len(stats["counters"][key])
                cur_row.append(imgs_cnt)
            pd_data.append(cur_row)

        pd_data[:0] = [columns]
        stats["pd_data"] = pd_data

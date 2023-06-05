from dataset_tools.convert import unpack_if_archive


def to_supervisely(input_path: str, output_path: str = None):
    input_dir = unpack_if_archive(input_path)
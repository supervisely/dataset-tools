import os
import zipfile
import tarfile


def unpack_if_archive(path: str) -> str:
    if os.path.isdir(path):
        return path

    extraction_path = os.path.splitext(path)[0]

    if zipfile.is_zipfile(path):
        os.makedirs(extraction_path, exist_ok=True)

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)

        return extraction_path

    if tarfile.is_tarfile(path):
        os.makedirs(extraction_path, exist_ok=True)

        with tarfile.open(path, "r") as tar_ref:
            tar_ref.extractall(extraction_path)

        return extraction_path

    return path

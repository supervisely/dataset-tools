import os
import tarfile
import zipfile

import tqdm

from supervisely.io.fs import get_file_name_with_ext


def unpack_if_archive(path: str) -> str:
    if os.path.isdir(path):
        return path

    extraction_path = os.path.splitext(path)[0]

    if zipfile.is_zipfile(path):
        os.makedirs(extraction_path, exist_ok=True)

        with zipfile.ZipFile(path, "r") as zip_ref:
            total_files = len(zip_ref.infolist())

            with tqdm.tqdm(
                desc=f"Unpacking '{get_file_name_with_ext(path)}'",
                total=total_files,
                unit="file",
            ) as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extraction_path)
                    pbar.update(1)

            return extraction_path

    if tarfile.is_tarfile(path):
        os.makedirs(extraction_path, exist_ok=True)

        with tarfile.open(path, "r") as tar_ref:
            total_files = len(tar_ref.getnames())

            with tqdm.tqdm(
                desc=f"Unpacking '{get_file_name_with_ext(path)}'",
                total=total_files,
                unit="file",
            ) as pbar:
                for file in tar_ref.getnames():
                    tar_ref.extract(file, extraction_path)
                    pbar.update(1)

            return extraction_path

    return path

import os
from pathlib import Path
import subprocess
from supervisely.sly_logger import logger


def convert_all(directory: str):
    current_dir = Path(__file__).parent.absolute()
    parent_dir = Path(__file__).parent.parent.parent.absolute()
    script_path = os.path.join(current_dir, "convert.sh")
    if not directory.startswith(str(parent_dir)):
        directory = os.path.join(parent_dir, directory)
    if not os.path.isdir(directory):
        raise("No such directory. Check the given path.")

    process = subprocess.run(
        ["bash", script_path, directory],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        process.check_returncode()
        logger.info(f"Files successfully converted.")
    except subprocess.CalledProcessError as e:
        print(e.stdout)

import json
import os
from typing import List
from urllib.parse import urljoin

import requests
import supervisely as sly
import tqdm
from supervisely.api.file_api import FileInfo

from dataset_tools.convert import unpack_if_archive

CURENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURENT_DIR)

PATH_DOWNLOAD_URLS = os.path.join(PARENT_DIR, "data/download_urls/released_datasets.json")


def prepare_link(
    api: sly.Api,
    project: sly.ProjectInfo,
    force: bool,
    tf_urls_path: str,
    files: List[FileInfo],
    clean_duplicates: bool = True,
):
    team_id = sly.env.team_id()
    workspace_id = sly.env.workspace_id()
    agent_id = sly.env.agent_id()
    storage_dir = sly.app.get_data_dir()
    local_save_path = os.path.join(storage_dir, "tmp/download_urls.json")

    # if os.path.exists(urls_path):
    #     with open(urls_path, "r") as f:
    #         urls = json.load(f)

    # api.project.update_custom_data(project_info.id, params_dtools)
    # sly.logger.info("Custom data updated with LICENSE.md and README.md contents.")

    if api.file.exists(team_id, tf_urls_path):
        api.file.download(team_id, tf_urls_path, local_save_path)
        with open(local_save_path, "r") as f:
            urls = json.load(f)
    else:
        keys = [project.name for project in api.project.get_list(workspace_id)]
        urls = {key: {} for key in keys}

    try:
        urls[project.name]
    except KeyError:
        raise KeyError(
            f"Download URL for dataset '{project.name}' not found. Please update dataset-tools to the latest version with 'pip install --upgrade dataset-tools'"
        )

    if not force:       
        if "https://www.dropbox.com" in urls[project.name]["download_sly_url"]:
            return urls[project.name]["download_sly_url"]

    def _get_duplicates(files):
        split_dict = {}
        for file in files:
            project_id = int(file.name.split("_")[1])
            if project_id not in split_dict:
                split_dict[project_id] = [file]
            else:
                split_dict[project_id].append(file)

        _to_delete_infos, _actual_links_infos = {}, {}
        for project_id, split_list in split_dict.items():
            sorted_ = sorted(split_list, key=lambda x: int(x.name.split("_")[1]))
            _actual_links_infos[project_id] = sorted_[-1]
            if len(split_list) >= 2:
                _to_delete_infos[project_id] = sorted_[:-1]
        return _to_delete_infos, _actual_links_infos

    to_delete_infos, actual_links_infos = _get_duplicates(files)
    if clean_duplicates is True:
        to_delete_paths = [item.path for sublist in to_delete_infos.values() for item in sublist]
        if len(to_delete_paths) > 0:
            pbar = tqdm.tqdm(desc="Deleting duplicate links", total=len(to_delete_paths))
            api.file.remove_batch(team_id, to_delete_paths, pbar)

    if not force:
        if actual_links_infos.get(project.id) is not None:
            if int(actual_links_infos[project.id].name.split("_")[1]) == project.id:
                sly.logger.info("URL already exists. Skipping creation of download link...")
                return actual_links_infos[project.id].full_storage_url

        sly.logger.info("URL not exists. Creating a download link...")
    else:
        sly.logger.info("Creating a download link...")

    app_slug = "supervisely-ecosystem/export-to-supervisely-format"
    module_id = api.app.get_ecosystem_module_id(app_slug)
    module_info = api.app.get_ecosystem_module_info(module_id)

    sly.logger.info(f"Start app: {module_info.name}")

    params = module_info.get_arguments(images_project=project.id)

    session = api.app.start(
        agent_id=agent_id,
        module_id=module_id,
        workspace_id=workspace_id,
        task_name="Prepare download link",
        params=params,
        app_version="dninja",
        is_branch=True,
    )
    sly.logger.info(f"Task started, task_id: {session.task_id}")
    sly.logger.info(session)

    try:
        # wait until task end or specific task status
        sly.logger.info("Waiting for the download link to finish being created...")
        api.app.wait(session.task_id, target_status=api.task.Status.FINISHED)

    except sly.WaitingTimeExceeded as e:
        sly.logger.error(e)
        # we don't want to wait more, let's stop our long-lived or "zombie" task
        api.app.stop(session.task_id)
    except sly.TaskFinishedWithError as e:
        sly.logger.error(e)

    # let's list all sessions of specific app in our team with additional optional filtering by statuses [finished]
    sessions = api.app.get_sessions(
        team_id=team_id, module_id=module_id, statuses=[api.task.Status.FINISHED]
    )
    return urljoin(
        os.environ["SERVER_ADDRESS"],
        sessions[0].details["meta"]["output"]["general"]["titleUrl"],
    )


def update_sly_url_dict(api: sly.Api, new_dict: dict, tf_urls_path: str) -> None:
    team_id = sly.env.team_id()

    storage_dir = sly.app.get_data_dir()
    local_save_path = os.path.join(storage_dir, "tmp/download_urls.json")

    if api.file.exists(team_id, tf_urls_path):
        api.file.download(team_id, tf_urls_path, local_save_path)
        with open(local_save_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    sly.logger.info("Updating dictionary with download links...")

    data.update(new_dict)

    with open(local_save_path, "w") as f:
        json.dump(data, f, indent=4)

    api.file.upload(team_id, local_save_path, tf_urls_path)
    sly.logger.info(f"Dictionary saved to Team files: '{tf_urls_path}'")


def download(dataset: str, dst_dir: str = "~/dataset-ninja/", unpack_archive: bool = True) -> str:
    dataset_name = dataset.lower().replace(" ", "-")

    dst_dir = os.path.expanduser(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    try:
        with open(PATH_DOWNLOAD_URLS, "r") as f:
            data = json.load(f)
    except Exception:
        raise FileNotFoundError(
            "File with download urls was not found. Please update dataset-tools to the latest version with 'pip install --upgrade dataset-tools'"
        )
    try:
        data[dataset]
    except KeyError:
        raise KeyError(
            f"Dataset '{dataset}' not found. Please check dataset name or update dataset-tools to the latest version with 'pip install --upgrade dataset-tools'"
        )

    sly_url = data[dataset]["download_sly_url"]

    response = requests.get(sly_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # Adjust the block size as needed

    dst_path = os.path.join(dst_dir, f"{dataset_name}.tar")

    with tqdm.tqdm(
        desc=f"Downloading '{dataset}'", total=total_size, unit="B", unit_scale=True
    ) as pbar:
        with open(dst_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                pbar.update(len(data))

    if unpack_archive:
        return unpack_if_archive(dst_path)
    return dst_path

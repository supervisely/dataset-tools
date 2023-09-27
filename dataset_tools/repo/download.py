import json
import os
from urllib.parse import urljoin

import requests
import supervisely as sly
import tqdm
from supervisely._utils import camel_to_snake

from dataset_tools.convert import unpack_if_archive

CURENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURENT_DIR)

PATH_DOWNLOAD_URLS = os.path.join(PARENT_DIR, "data/download_urls/released_datasets.json")


def prepare_link(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    force: bool,
    tf_urls_path: str,
    params_dtools: dict = None,
):
    team_id = sly.env.team_id()
    workspace_id = sly.env.workspace_id()
    agent_id = sly.env.agent_id()
    storage_dir = sly.app.get_data_dir()
    local_save_path = os.path.join(storage_dir, "tmp/download_urls.json")

    # if os.path.exists(urls_path):
    #     with open(urls_path, "r") as f:
    #         urls = json.load(f)

    api.project.update_custom_data(project_info.id, params_dtools)
    sly.logger.info("Custom data updated with LICENSE.md and README.md contents.")

    if api.file.exists(team_id, tf_urls_path):
        api.file.download(team_id, tf_urls_path, local_save_path)
        with open(local_save_path, "r") as f:
            urls = json.load(f)
    else:
        keys = [project.name for project in api.project.get_list(workspace_id)]
        urls = {key: {} for key in keys}

    # try:
    #     urls[project_info.name]
    # except KeyError:
    #     raise KeyError(
    #         f"Download URL for dataset '{project_info.name}' not found. Please update dataset-tools to the latest version with 'pip install --upgrade dataset-tools'"
    #     )

    if not force:
        if (
            urls.get(project_info.name) is not None
            and urls[project_info.name].get("id") == project_info.id
            and urls[project_info.name].get("download_sly_url") is not None
        ):
            sly.logger.info("URL already exists. Skipping creation of download link...")
            return urls[project_info.name]["download_sly_url"]

        sly.logger.info("URL not exists. Creating a download link...")
    else:
        sly.logger.info("Creating a download link...")

    app_slug = "supervisely-ecosystem/export-to-supervisely-format"
    module_id = api.app.get_ecosystem_module_id(app_slug)
    module_info = api.app.get_ecosystem_module_info(module_id)

    sly.logger.info(f"Start app: {module_info.name}")

    params = module_info.get_arguments(images_project=project_info.id)

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

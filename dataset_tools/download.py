import json
import os
from urllib.parse import urljoin

import supervisely as sly


def prepare_download_link(project):
    api = sly.Api.from_env()

    team_id = sly.env.team_id()
    workspace_id = sly.env.workspace_id()
    agent_id = sly.env.agent_id()
    storage_dir = sly.app.get_data_dir()
    local_save_path = os.path.join(storage_dir, "download_links.json")
    api.file.download(team_id, os.environ["DOWNLOADS_DICT"], local_save_path)
    with open(local_save_path, "r") as f:
        links = json.load(f)

    if links.get(str(project.id), None) is not None:
        print("URL already exists. Skipping creation of download link...")
        return links[str(project.id)]
    else:
        print("URL not exists. Creating a download link...")

        app_slug = f"supervisely-ecosystem/export-to-supervisely-format"
        module_id = api.app.get_ecosystem_module_id(app_slug)
        module_info = api.app.get_ecosystem_module_info(module_id)

        print("Start app: ", module_info.name)

        params = module_info.get_arguments(images_project=project.id)

        session = api.app.start(
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=workspace_id,
            task_name="custom session name",
            params=params,
        )
        print("App is started, task_id =", session.task_id)
        print(session)

        try:
            # wait until task end or specific task status
            print("Waiting for the download link to finish being created...")
            api.app.wait(session.task_id, target_status=api.task.Status.FINISHED)

        except sly.WaitingTimeExceeded as e:
            print(e)
            # we don't want to wait more, let's stop our long-lived or "zombie" task
            api.app.stop(session.task_id)
        except sly.TaskFinishedWithError as e:
            print(e)

        # let's list all sessions of specific app in our team with additional optional filtering by statuses [finished]
        sessions = api.app.get_sessions(
            team_id=team_id, module_id=module_id, statuses=[api.task.Status.FINISHED]
        )
        return urljoin(
            os.environ["SERVER_ADDRESS"],
            sessions[0].details["meta"]["output"]["general"]["titleUrl"],
        )


def update_sly_url_dict(new_dict: dict) -> None:
    src = os.environ["DOWNLOADS_DICT"]
    dst = os.path.join(sly.app.get_data_dir(), "download_links.json")

    print("Updating dictionary with download links...")
    api = sly.Api.from_env()
    team_id = sly.env.team_id()

    if api.file.exists(team_id, src):
        api.file.download(team_id, src, dst)

        with open(dst, "r") as f:
            data = json.load(f)
        data.update(new_dict)
    else:
        data = new_dict

    with open(dst, "w") as f:
        json.dump(data, f)

    api.file.upload(team_id, src=dst, dst=src)

    print("Dictionary successfully updated!")

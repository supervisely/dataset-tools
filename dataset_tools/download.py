import os
import re
import json

from dotenv import load_dotenv
from tqdm import tqdm

import supervisely as sly


# if sly.is_development():
#     load_dotenv(os.path.expanduser("~/ninja.env"))
#     load_dotenv("local.env")


# data_dir = sly.app.get_data_dir()
# # task_id = sly.env.task_id()
# team_id = sly.env.team_id()

# batch_size = 10

# team_id = sly.env.team_id()
# workspace_id = sly.env.workspace_id()
# # agent_id = sly.env.agent_id()
# agent_id = 1
# project_id = sly.env.project_id()


def prepare_download_link(project):
    api = sly.Api.from_env()

    app_slug = f"supervisely-ecosystem/export-to-supervisely-format"

    module_id = api.app.get_ecosystem_module_id(app_slug)
    # module_id = 83  # or copy module ID of application in ecosystem
    module_info = api.app.get_ecosystem_module_info(module_id)
    print("Start app: ", module_info.name)

    print("List of available app arguments for developers (like --help in terminal):")
    module_info.arguments_help()

    params = module_info.get_arguments(images_project=project.id)

    session = api.app.start(
        agent_id=agent_id,
        module_id=module_id,
        workspace_id=workspace_id,
        task_name="custom session name",
        params=params,
    )
    print("App is started, task_id = ", session.task_id)
    print(session)

    try:
        # wait until task end or specific task status
        api.app.wait(session.task_id, target_status=api.task.Status.FINISHED)

    except sly.WaitingTimeExceeded as e:
        print(e)
        # we don't want to wait more, let's stop our long-lived or "zombie" task
        api.app.stop(session.task_id)
    except sly.TaskFinishedWithError as e:
        print(e)

    print("Task status: ", api.app.get_status(session.task_id))

    # let's list all sessions of specific app in our team with additional optional filtering by statuses [finished]
    sessions = api.app.get_sessions(
        team_id=team_id, module_id=module_id, statuses=[api.task.Status.FINISHED]
    )
    for session_info in sessions:
        print(session_info)


# def prepare_download_link(project_id):
#     project = api.project.get_info_by_id(project_id)
#     datasets = api.dataset.get_list(project.id)
#     dataset_ids = [dataset.id for dataset in datasets]

#     download_json_plus_images(api, project, dataset_ids)

#     download_dir = os.path.join(data_dir, f"{project.id}_{project.name}")
#     full_archive_name = str(str(project.id) + "_" + project.name + ".tar")
#     result_archive = os.path.join(data_dir, full_archive_name)
#     sly.fs.archive_directory(download_dir, result_archive)

#     remote_archive_path = os.path.join(
#         sly.team_files.RECOMMENDED_EXPORT_PATH,
#         "export-to-supervisely-format/{}".format(full_archive_name),
#     )


#     with tqdm(desc=f"Upload {full_archive_name}", total=int(project.size)) as pbar:
#         file_info = api.file.upload(team_id, result_archive, remote_archive_path, progress_cb=pbar)

#     print("Uploaded to Team-Files: {!r}".format(file_info.storage_path))

#     return os.path.join(os.environ["SERVER_ADDRESS"], file_info.storage_path)

#     # api.task.set_output_archive(
#     #     task_id, file_info.id, full_archive_name, file_url=file_info.storage_path
#     # )


# def download_json_plus_images(api, project, dataset_ids):
#     download_dir = os.path.join(data_dir, f"{project.id}_{project.name}")

#     with tqdm(desc=f"Download {project.name}", total=project.items_count) as pbar:
#         sly.download_project(
#             api,
#             project.id,
#             download_dir,
#             dataset_ids=dataset_ids,
#             progress_cb=pbar,
#             batch_size=batch_size,
#         )
#     print("Project {!r} has been successfully downloaded.".format(project.name))


def update_links_dict(dct: dict):
    pass

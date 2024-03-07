from datetime import datetime
import json
import supervisely as sly
from pathlib import Path
import os

TEAM_ID = 9
WORKSPACE_ID = 28


def generate_repos_list():
    api = sly.Api.from_env(env_file=os.path.join(Path.home(), "ninja.env"))
    json_path = "tests/repos.json"
    # api.file.download(TEAM_ID, "/ninja-updater/_repos_list/repos.json", json_path)

    projects = api.project.get_list(WORKSPACE_ID)
    repos = {}

    for project in projects:
        dt = datetime.strptime(project.created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        month = dt.strftime("%B")
        year = dt.strftime("%Y")
        mmyy = f"{month}-{year}"

        github_url = project.custom_data.get("github_url")
        ds_is_hidden = project.custom_data.get("hide_dataset")

        if github_url is not None and ds_is_hidden is False:
            try:
                repos[mmyy].append(github_url)
            except KeyError:
                repos[mmyy] = [github_url]
            try:
                repos[mmyy].index(github_url)
            except ValueError:
                repos[mmyy].append(github_url)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(repos, f)

    api.file.upload(TEAM_ID, json_path, "/ninja-updater/_repos_list/repos.json")


if __name__ == "__main__":
    generate_repos_list()

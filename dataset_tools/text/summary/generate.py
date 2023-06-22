import operator
import os
import re
import textwrap
from typing import Dict, List, Optional, Union

import inflect
import supervisely as sly

p = (
    inflect.engine()
)  # correctly generate plurals, singular nouns, ordinals, indefinite articles; convert numbers to words.


def list2sentence(
    lst: Union[List[str], str],
    anytail: str = "",
    keeptail=False,
    url: Optional[Union[List[str], str]] = None,
    char2wrap: Optional[str] = None,
):
    if isinstance(lst, str):
        lst = [lst]
    if isinstance(url, str):
        url = [url]
    assert isinstance(lst, list) and all(
        isinstance(item, str) for item in lst
    ), "All items in the list must be strings."

    anytail = " " + anytail if anytail != "" else anytail
    if len(lst) == 0:
        raise ValueError("Provided list is empty")

    if url is not None:
        new_lst = []
        for i, u in zip(lst, url):
            new_lst.append(f"[{i}]({u})")
        lst = new_lst

    if char2wrap is not None:
        lst = [char2wrap + elem + char2wrap for elem in lst]

    if len(lst) == 1:
        sentence = lst[0]
    elif len(lst) == 2:
        sentence = " and ".join(lst)
    else:
        sentence = ", ".join(lst[:-1]) + ", and " + lst[-1]

    if keeptail:
        return sentence + anytail

    if anytail != "":
        if p.singular_noun(anytail):
            sentence += anytail if len(lst) > 1 else p.singular_noun(anytail)
        else:
            sentence += p.plural_noun(anytail) if len(lst) > 1 else anytail

    return sentence.strip()


def standardize(text: str):
    return re.sub(r"[_-]", " ", text).strip()


def get_summary_data(
    name: str,
    fullname: str,
    cv_tasks: List[str],
    annotation_types: List[str],
    industries: str,
    release_year: int,
    homepage_url: str,
    license: str,
    license_url: str,
    preview_image_id: int,
    github_url: str,
    download_sly_url: str,
    download_original_url: str = None,
    paper: str = None,
    citation_url: str = None,
    organization_name: str = None,
    organization_url: str = None,
    tags: List[str] = None,
    **kwargs,
) -> Dict:
    api = sly.Api.from_env()
    project_id = kwargs.get("project_id", None)
    project_info = api.project.get_info_by_id(project_id)

    stats = api.project.get_stats(project_id)

    notsorted = [
        [cls["objectClass"]["name"], cls["total"]] for cls in stats["images"]["objectClasses"]
    ]
    totals_dct = {
        "total_assets": stats["images"]["total"]["imagesInDataset"],
        "total_objects": stats["objects"]["total"]["objectsInDataset"],
        "total_classes": len(stats["images"]["objectClasses"]),
        "top_classes": list(
            map(operator.itemgetter(0), sorted(notsorted, key=operator.itemgetter(1), reverse=True))
        ),
    }

    unlabeled_num = stats["images"]["total"]["imagesNotMarked"]
    unlabeled_percent = round(unlabeled_num / totals_dct["total_assets"] * 100)

    splits_list = [
        {"name": item["name"], "split_size": item["imagesCount"]}
        for item in stats["datasets"]["items"]
    ]

    fields = {
        # preset fields
        "name": name,
        "fullname": fullname,
        "cv_tasks": cv_tasks,
        "annotation_types": annotation_types,
        "industries": industries,
        "release_year": release_year,
        "homepage_url": homepage_url,
        "license": license,
        "license_url": license_url,
        "preview_image_id": preview_image_id,
        "github_url": github_url,
        "citation_url": citation_url,
        "download_sly_url": download_sly_url,
        # from supervisely
        "modality": project_info.type,
        "totals": totals_dct,
        "unlabeled_assets_num": unlabeled_num,
        "unlabeled_assets_percent": unlabeled_percent,
        "splits": splits_list,
    }

    # optional fields
    for key, value in zip(
        ["download_original_url", "paper", "organization_name", "organization_url", "tags"],
        [download_original_url, paper, organization_name, organization_url, tags],
    ):
        if value is not None:
            fields[key] = value

    return fields


def generate_summary_content(data: Dict, vis_url: str = None) -> str:
    # preset fields
    # required
    name = data.get("name", None)
    fullname = data.get("fullname", None)
    cv_tasks = [standardize(cv_task) for cv_task in data.get("cv_tasks", [])]
    annotation_types = [standardize(ann_type) for ann_type in data.get("annotation_types", [])]
    industries = data.get("industries", None)
    release_year = data.get("release_year", None)
    homepage_url = data.get("homepage_url", None)
    license = data.get("license", None)
    license_url = data.get("license_url", None)
    preview_image_id = data.get("preview_image_id", None)
    github_url = data.get("github_url", None)
    citation_url = data.get("citation_url", None)
    download_sly_url = data.get("download_sly_url", None)

    # optional
    download_original_url = data.get("download_original_url", None)
    paper = data.get("paper", None)
    organization_name = data.get("organization_name", None)
    organization_url = data.get("organization_url", None)
    tags = (data.get("tags", []),)

    # from supervisely
    # required
    modality = data.get("modality")
    totals = data.get("totals", {})
    top_classes = totals.get("top_classes", [])
    unlabeled_assets_num = data.get("unlabeled_assets_num", None)
    unlabeled_assets_percent = data.get("unlabeled_assets_percent", None)
    splits = [
        f'*{split["name"]}* ({split["split_size"]} {modality})' for split in data.get("splits", [])
    ]

    # prepare data
    annotations = []
    if "instance segmentation" in annotation_types:
        if (
            "semantic segmentation" not in annotation_types
            and "object detection" not in annotation_types
        ):
            annotations.append(
                " pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks"
            )
        elif (
            "semantic segmentation" in annotation_types
            and "object detection" not in annotation_types
        ):
            annotations.append(
                " pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into an object detection (bounding boxes for every object) task"
            )
        elif (
            "semantic segmentation" not in annotation_types
            and "object detection" in annotation_types
        ):
            annotations.append(
                " pixel-level instance segmentation and bounding box annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation task (only one mask for every class)"
            )
        else:
            annotations.append(" pixel-level instance segmentation annotations")
    else:
        if "semantic segmentation" in annotation_types:
            annotations.append(
                " pixel-level semantic segmentation annotations. Due to the nature of the semantic segmentation task, it can be automatically transformed into object detection (bounding boxes for every object) task"
            )
        elif "object detection" in annotation_types:
            annotations.append(" bounding box annotations")

    annotations = ",".join(annotations).strip()

    # collect content
    content = f"**{fullname}**"
    # if fullname is not None:
    #    content += f" ({fullname})"
    content += f" is a dataset for {list2sentence(cv_tasks, 'tasks', keeptail=True)}. "

    if "general domain" in industries:
        content += "It is applicable or relevant across various domains."
        if len(industries) > 1:
            industries.pop("general domain")
            content += f"Also, it is used in {list2sentence(industries, 'industries')}."
    else:
        content += f"It is used in the {list2sentence(industries, 'industries')}."

    content += "\n\n"
    content += f"The dataset consists of {totals.get('total_assets', 0)} {modality} with {totals.get('total_objects', 0)} labeled objects belonging to {totals.get('total_classes', 0)} "
    if len(top_classes) == 1:
        content += f"single class "
    else:
        content += f"different classes "
    if len(top_classes) > 3:
        content += f"including *{'*, *'.join(top_classes[:3])}*, and other: {list2sentence(top_classes[3:], char2wrap='*' )}."
    elif len(top_classes) == 1:
        content += f"(*{top_classes[0]}*)."
    else:
        content += f"including {list2sentence(top_classes[:3], char2wrap='*')}."
    content += f"\n\nEach {p.singular_noun(modality)} in the {name} dataset has {annotations}. "
    if unlabeled_assets_num == 0:
        content += f"All {modality} are labeled (i.e. with annotations). "
    elif unlabeled_assets_num == 1:
        content += f"There is 1 unlabled {p.singular_noun(modality)} (i.e. without annotations). "
    else:
        content += f"There are {unlabeled_assets_num} ({unlabeled_assets_percent}% of the total) unlabeled {modality} (i.e. without annotations). "
    if len(splits) == 1:
        content += f"There is 1 split in the dataset: {list2sentence(splits)}. "
    else:
        content += f"There are {len(splits)} splits in the dataset: {list2sentence(splits)}. "
    if organization_name is not None and organization_url is not None:
        content += f"The dataset was released in {release_year} by the {list2sentence(organization_name, url=organization_url)}."
    elif organization_name is not None and organization_url is None:
        content += (
            f"The dataset was released in {release_year} by the {list2sentence(organization_name)}."
        )
    elif organization_name is None and organization_url is None:
        content += f"The dataset was released in {release_year}."

    if vis_url is not None and isinstance(vis_url, str):
        if vis_url.endswith("poster.png"):
            content += f'\n\n<img src="{vis_url}">\n'
        elif vis_url.endswith(".png"):
            content += f"\n\nHere is the visualized example grid with annotations:\n\n"
            content += f'<img src="{vis_url}">\n'
        else:
            content += f"\n\nHere are the visualized examples for each of the {totals.get('total_classes', 0)} classes:\n\n"
            content += f"[Dataset classes]({vis_url})\n"

    return content


def get_summary_data_sly(project_info: sly.ProjectInfo) -> Dict:
    return get_summary_data(**project_info.custom_data, project_id=project_info.id)


# def generate_meta_from_local():

#     modality ="images"

#     with open("./stats/class_balance.json") as f:
#         json_data = json.load(f)
#     df = pd.DataFrame(data=json_data["data"], columns=json_data["columns"])

#     with open("./stats/classes_per_image.json") as f:
#         json_data = json.load(f)
#     df_img = pd.DataFrame(data=json_data["data"], columns=json_data["columns"])


#     totals_dct = {
#        "total_modality_files": df_img.shape[0],
#        "total_objects": df["Objects"].sum(),
#        "total_classes": df["Class"].count(),
#        "top_classes": df.sort_values('Objects', ascending=False)['Class'].tolist()
#     }

#     unlabeled_num = df_img.shape[0] - sum(df_img.drop(columns=["Image","Split", "Height", "Width", "Unlabeled"]).sum(axis=1)==0)
#     unlabeled_percent = unlabeled_num / df_img.shape[0]
#     splits_list = [
#         {
#         "name": "training",
#         "split_size": 800
#         },
#         {
#         "name": "validation",
#         "split_size": 200
#         }
#     ]

#     return {
#         "name": "PASCAL VOC",
#         "fullname": "PASCAL Visual Object Classes Challenge",
#         "cv_tasks": ["semantic-segmentation"],
#         "modality": modality,
#         "release_year": "2012",
#         "organization": "Dong et al",
#         "organization_link": "https://arxiv.org/pdf/2012.07131v2.pdf",
#         "totals": totals_dct,
#         "unlabeled_assets_num": unlabeled_num,
#         "unlabeled_assets_percent": unlabeled_percent,
#         "splits": splits_list
#     }

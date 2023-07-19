import itertools
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

MAX_CLASSES_IN_TEXT = 25


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
    assert (
        isinstance(lst, list)
        or isinstance(lst, tuple)
        and all(isinstance(item, str) for item in lst)
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
    elif len(lst) > MAX_CLASSES_IN_TEXT * 2:
        sentence = (
            ", ".join(lst[:MAX_CLASSES_IN_TEXT])
            + ", and "
            + str(len(lst) - MAX_CLASSES_IN_TEXT)
            + " more"
        )
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
    applications: str,
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
    slytagsplit: Dict[str, List[str]] = None,
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

    slydssplits_list = [
        {"name": item["name"], "split_size": item["imagesCount"]}
        for item in stats["datasets"]["items"]
    ]

    slytagsplits_dict = {}
    if slytagsplit is not None:
        for group_name, slytag_names in slytagsplit.items():
            slytagsplits_dict[group_name] = [
                {
                    "name": item["tagMeta"]["name"],
                    "split_size": item["total"],
                    "datasets": item["datasets"],
                }
                for item in stats["imageTags"]["items"]
                if item["tagMeta"]["name"] in slytag_names
            ]

    splits = {
        "slyds": slydssplits_list,
        "slytag": slytagsplits_dict,
    }

    fields = {
        # preset fields
        "name": name,
        "fullname": fullname,
        "cv_tasks": cv_tasks,
        "annotation_types": annotation_types,
        "applications": applications,
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
        "splits": splits,
    }

    # backward compatibility
    if len(applications) == 0:
        fields["industries"] = kwargs.get("industries")

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

    # backward compatibility
    if data.get("industries") is not None:
        sly.logger.warn("Field 'INDUSTRIES' is deprecated. Please use 'APPLICATIONS' instead")
        industries = data["industries"]
    else:
        sorted_ = sorted(data.get("applications"), key=lambda x: x["is_used"])
        grouped_ = itertools.groupby(sorted_, key=lambda x: x["is_used"])
        applications = {}
        for key, group in grouped_:
            postfix_group = itertools.groupby(group, key=lambda x: x["postfix"])
            applications[key] = {
                postfix: [elem["text"] for elem in postfix_group]
                for postfix, postfix_group in postfix_group
            }

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

    slyds_splits = [
        f'*{split["name"]}* ({split["split_size"]} {modality})'
        for split in data["splits"].get("slyds", [])
    ]

    slytag_splits = {}
    for group_name, splits in data["splits"].get("slytag", {}).items():
        # extras = [[(ds["name"], ds["count"]) for ds in split["datasets"]] for split in splits]
        # for extra in extras:
        #     slyds, count = extra
        #     arr = [f"[i]{s},{c}[/i]" for s, c in zip(slyds, count)]

        slytag_splits[group_name] = [
            f'*{split["name"]}* ({split["split_size"]} {modality})' for split in splits
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
                " pixel-level semantic segmentation annotations. Due to the nature of the semantic segmentation task, it can be automatically transformed into an object detection (bounding boxes for every object) task"
            )
        elif "object detection" in annotation_types:
            annotations.append(" bounding box annotations")

    annotations = ",".join(annotations).strip()

    content = f"**{fullname}**"

    content += f" is a dataset for {list2sentence(cv_tasks, 'tasks', keeptail=True)}. "

    # backward compatibility
    if data.get("industries") is not None:
        if "general" in industries:
            content += "It is applicable or relevant across various domains."
            if len(industries) > 1:
                industries.pop("general domain")
                content += f"Also, it is used in {list2sentence(industries, 'industries')}."
        else:
            content += f"It is used in the {list2sentence(industries, 'industries')}."
    else:
        if (
            applications.get(True) is not None
            and applications[True].get("domain") is not None
            and "general" in applications[True]["domain"]
        ):
            content += "It is applicable or relevant across various domains. "
            extended = [item for sublist in applications[True].values() for item in sublist]
            if len(extended) > 1:
                applications[True]["domain"].remove("general")
                content += f"Also, it is used in the "
                tmp_list = []
                for postfix, text in applications[True].items():
                    if text:
                        tmp_list.append(f"{list2sentence(text, postfix)}")
                content += list2sentence(tmp_list) + ". "
        else:
            if applications.get(True) is not None:
                content += "It is used in the "
                tmp_list = []
                for postfix, text in applications[True].items():
                    tmp_list.append(f"{list2sentence(text, postfix)}")
                content += list2sentence(tmp_list) + ". "

        if applications.get(False) is not None:
            content += "Possible applications of the dataset could be in the "
            tmp_list = []
            for postfix, text in applications[False].items():
                tmp_list.append(f"{list2sentence(text, postfix)}")
            content += list2sentence(tmp_list) + ". "

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
    content += f"\n\n{modality.capitalize()} in the {name} dataset have {annotations}. "
    if unlabeled_assets_num == 0:
        content += f"All {modality} are labeled (i.e. with annotations). "
    elif unlabeled_assets_num == 1:
        content += f"There is 1 unlabeled {p.singular_noun(modality)} (i.e. without annotations). "
    else:
        content += f"There are {unlabeled_assets_num} ({unlabeled_assets_percent}% of the total) unlabeled {modality} (i.e. without annotations). "

    if len(slyds_splits) == 1:
        content += f"There are no pre-defined <i>train/val/test</i> splits in the dataset. "
    else:
        content += (
            f"There are {len(slyds_splits)} splits in the dataset: {list2sentence(slyds_splits)}. "
        )

    if len(slytag_splits) > 0:
        content += f"Alternatively, dataset could be splitted by "
        for idx, items in enumerate(slytag_splits.items()):
            if idx == 0:
                content += f"<i>{items[0]}</i> "
            else:
                content += f", or <i>{items[0]}</i> "
            content += f"criteria: {list2sentence(items[1])}"
        content += ". "

    if organization_name is not None and organization_url is not None:
        content += f"The dataset was released in {release_year} by the {list2sentence(organization_name, url=organization_url)}."
    elif organization_name is not None and organization_url is None:
        content += (
            f"The dataset was released in {release_year} by the {list2sentence(organization_name)}."
        )
    elif organization_name is None and organization_url is None:
        content += f"The dataset was released in {release_year}."

    total_classes = totals.get("total_classes", 0)
    if vis_url is not None and isinstance(vis_url, str):
        if vis_url.endswith("poster.png"):
            content += f'\n\n<img src="{vis_url}">\n'
        elif vis_url.endswith(".png"):
            content += f"\n\nHere is the visualized example grid with annotations:\n\n"
            content += f'<img src="{vis_url}">\n'
        elif total_classes > MAX_CLASSES_IN_TEXT:
            content += f"\n\nHere is a visualized example for {MAX_CLASSES_IN_TEXT} randomly selected sample classes:\n\n"
            content += f"[Dataset classes]({vis_url})\n"
        else:
            content += (
                f"\n\nHere are the visualized examples for each of the {total_classes} classes:\n\n"
            )
            content += f"[Dataset classes]({vis_url})\n"

    return content


def get_summary_data_sly(project_info: sly.ProjectInfo) -> Dict:
    return get_summary_data(**project_info.custom_data, project_id=project_info.id)

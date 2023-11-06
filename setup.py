import os
import re
import requests
from pkg_resources import DistributionNotFound, get_distribution

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


response = requests.get("https://api.github.com/repos/supervisely/dataset-tools/releases/latest")
version = response.json()["tag_name"]


INSTALL_REQUIRES = [
    "supervisely>=6.72.28",
    "numpy>=1.19, <2.0.0",
    "requests>=2.27.1, <3.0.0",
    "requests-toolbelt>=0.9.1, <1.0.0",
    "tqdm>=4.62.3, <5.0.0",
    "pandas>=1.1.3, <=1.5.2",  # For compatibility with Python3.7
    "matplotlib>=3.3.2, <4.0.0",
    "scikit-image>=0.17.1, <1.0.0",
    "dataframe_image>=0.1.11, <1.0.0",
    "inflect>=6.0.0",
    "gdown>=4.7.1",
    "urllib3==1.26.15",
    "geojson>=3.0.0",
    "titlecase==2.4",
    "pycocotools>=2.0.0",
    "memory-profiler==0.61.0",
    "Pympler==1.0.1",
    "xmltodict==0.13.0",
    "imagesize==1.4.1",
]

ALT_INSTALL_REQUIRES = {
    # "opencv-python>=4.5.5.62, <5.0.0.0": [
    #     "opencv-python-headless",
    #     "opencv-contrib-python",
    #     "opencv-contrib-python-headless",
    # ],
}


def check_alternative_installation(install_require, alternative_install_requires):
    """If some version version of alternative requirement installed, return alternative,
    else return main.
    """
    for alternative_install_require in alternative_install_requires:
        try:
            alternative_pkg_name = re.split(r"[ !<>=]", alternative_install_require)[0]
            get_distribution(alternative_pkg_name)
            return str(alternative_install_require)
        except DistributionNotFound:
            continue

    return str(install_require)


def get_install_requirements(main_requires, alternative_requires):
    """Iterates over all install requires
    If an install require has an alternative option, check if this option is installed
    If that is the case, replace the install require by the alternative to not install dual package
    """
    install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires:
            main_require = check_alternative_installation(
                main_require, alternative_requires.get(main_require)
            )
        install_requires.append(main_require)

    return install_requires


# Dependencies do not include PyTorch, so
# supervisely_lib.nn.hosted.pytorch will not work out of the box.
# If you need to invoke that part of the code, it is very likely you
# already have PyTorch installed.
setup(
    name="dataset-tools",
    version=version,
    description="Dataset tools for dataset ninja made by Supervisely team.",
    packages=find_packages(include=["dataset_tools", "dataset_tools.*"]),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Supervisely",
    author_email="support@supervise.ly",
    url="https://github.com/supervisely/dataset-tools",
    package_data={
        "dataset_tools": ["data/*/*.json", "fonts/*.ttf"],
    },
    # entry_points={
    #     "console_scripts": [
    #         "sly-release=supervisely.release.run:cli_run",
    #         "supervisely=supervisely.cli.cli:cli",
    #     ]
    # },
    python_requires=">=3.7.1",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=get_install_requirements(INSTALL_REQUIRES, ALT_INSTALL_REQUIRES),
    extras_require={
        "docs": [
            "sphinx==4.4.0",
            "jinja2==3.0.3",
            "sphinx-immaterial==0.4.0",
            "sphinx-copybutton==0.4.0",
            "sphinx-autodoc-typehints==1.15.3",
            "sphinxcontrib-details-directive==0.1.0",
            "myst-parser==0.18.0",
        ],
    },
)

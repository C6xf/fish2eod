#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup script. Imports description from readme."""

import setuptools
import yaml
from fish2eod._version import __version__
from itertools import chain

def strip_dep(s):
    return s.split('--')[0].strip(' ')


def clean_dep(dep):
    if isinstance(dep, str):
        return [strip_dep(dep)]
    else:
        return [strip_dep(d) for d in list(dep.values())[0]]


with open("README.rst", "r") as fh:
    long_description = fh.read()


with open("conda_environment.yml") as f:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    envt = yaml.load(f, Loader=yaml.FullLoader)
    deps = envt['dependencies']
    clean_deps = list(chain(*map(clean_dep, deps)))

setuptools.setup(
    name="fish2eod",
    version=__version__,
    author="Aaron R. Shifman",
    author_email="ashifman@uottawa.ca",
    description="Automated simulation and inference of electric fields from weakly electric fish",
    long_description=long_description,
    long_description_content_type="text/rst",
    url="https://github.com/aaronshifman/fish2eod",
    packages=setuptools.find_packages(),
    setup_requires=clean_deps+["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
)

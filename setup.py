#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup script. Imports description from readme."""

import setuptools
import fish2eod

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fish2eod",
    version=fish2eod.__version__,
    author="Aaron R. Shifman",
    author_email="ashifman@uottawa.ca",
    description="Blah blah blah fem",
    long_description=long_description,
    long_description_content_type="text/rst",
    url="https://github.com/aaronshifman/fish2eod",
    install_requires=[],
    packages=setuptools.find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
)

import yaml

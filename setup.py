#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="spectrascope",
    version="0.1.0",
    license="Apache-2.0",
    description="We provide a toolkit for generating features for given spectrum data and build machine learning models",
    long_description="%s" % (read("README.md")),
    author="Amalya Johnson",
    author_email="amalya.johnson@tri.global",
    url="https://github.com/TRI-AMDD/spectrascope",
    packages=find_packages("spectrascope"),
    package_dir={"": "spectrascope"},
    py_modules=[splitext(basename(path))[0] for path in glob("spectrascope/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://mmmm.readthedocs.io/",
        "Changelog": "https://mmmm.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/TRI-AMDD/spectrascope/issues",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires="!=3.0.*, !=3.1.*, !=3.2.*, !=3.11.*, !=3.4.*",
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        # 'console_scripts': [
        #     'mmmm = mmmm.cli:main',
        # ]
    },
)

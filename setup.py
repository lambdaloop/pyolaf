#!/usr/bin/env ipython

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyolaf",
    version="0.1.0",
    author="Lili Karashchuk",
    author_email="krchtchk@gmail.com",
    description="3D reconstruction framework for light field microscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lambdaloop/pyolaf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    install_requires=[
        'pyyaml',
        'numpy',
        'scipy',
        'tqdm',
        'tifffile',
        'scikit-image'
    ],
    extras_require={
        'gpu':  ["cupy"]
    }

)

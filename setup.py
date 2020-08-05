"""This file contains rudimentary setup code for the project"""

import setuptools

with open("README.md", "r") as rm:
    long_description = rm.read()

setuptools.setup(
    name="Generic_GNN",
    version="0.0.1",
    author="Anonymous",
    author_email="flawnsontong1@gmail.com",
    description="Amino acid vae",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

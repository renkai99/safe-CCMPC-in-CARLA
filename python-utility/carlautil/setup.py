import os
from setuptools import setup, find_packages

install_requires=[
        'numpy',
        'pytest',
        'networkx']

immediate_dirpath = os.path.dirname(os.path.abspath(__file__))
readme_filepath = os.path.join(immediate_dirpath, "README.md")
with open(readme_filepath, "r") as fh:
    long_description = fh.read()

setup(
    name="carlautil",
    version="0.0.1",
    author="fireofearth",
    author_email="colinc@fastmail.com",
    description="utility methods for the CARLA simulator's Python API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
)

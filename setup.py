import os
import sys
from setuptools import find_packages, setup


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "socnavgym"))

NAME = "socnavgym"
DESCRIPTION = "An environment for Social Navigation"
REPOSITORY = "https://github.com/gnns4hri/SocNavGym"
EMAIL = "ljmanso@ljmanso.com"
AUTHOR = "Pilar Bachiller, Luis J. Manso, Sushant Swamy, Aditia Kapoor"
VERSION = "0.2.0"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

REQUIRED = [
    "gymnasium >= 0.29.1",
    "opencv-python",
    "numpy",
    "matplotlib",
    "shapely",
    "Cython"
]
EXCLUDES=["environment_configs", "tests"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=REPOSITORY,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=EXCLUDES),
    install_requires=REQUIRED,
    license="GPL-3",
    include_package_data=True
)


#!/usr/bin/env python

from setuptools import find_packages, setup
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()


setup(
    name="pyqsar3",
    version="1.4.3",
    description="SSU Lab Package for PYQSAR",
    author="SSU@Chemoinformatic Lab",
    author_email="chokh@ssu.ac.kr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chemoinfomatics/pyqsar3",
    packages=find_packages(),
)

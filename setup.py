
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='catalysis',
    version='0.1.0',

    description='Python tools for interacting with connectomics data from CATMAID',
    long_description=long_description,
    url='https://github.com/ceesem/catalysis',
    author='Casey Schneider-Mizell',
    author_email='caseysm@gmail.com',
    packages=['catalysis'],
    install_requires=requirements,
    include_package_data=True,
    data_files=[('data', ['data/Brain_Lineage_Landmarks_EMtoEM_ProjectSpace.csv',
                          'data/example_project_info.json',
                          'smat_jefferis.csv'])],

)

from pathlib import Path
from setuptools import setup, find_packages
import asid


setup(
    name='asid',
    version=asid.version,
    author='Ekaterina',
    long_description=open(Path(__file__).parent / 'README.md').read(),
    packages=find_packages(),
    test_suite='tests',
    install_requires=[],
)

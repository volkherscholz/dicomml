from setuptools import setup, find_packages

setup(
    # Application name:
    name="dicomml",

    # Version number:
    version="0.1.0",

    # Application author details:
    author="Volkher Scholz",
    author_email="volkher.scholz@gmail.com",

    # Packages
    packages=find_packages(exclude=['examples', 'tests*', 'k8s']),

    # Dependencies
    install_requires=['numpy',
                      'absl-py'],

    # Details
    url="https://github.com/volkherscholz/dicomml",

    #
    # license="LICENSE.txt",
    description="Dicom Machine Learning",

    long_description=open("README.md").read(),
)

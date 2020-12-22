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
    packages=find_packages(exclude=['examples', 'tests*']),
    # Dependencies
    install_requires=['numpy',
                      'pydicom',
                      'pandas',
                      'scipy',
                      'scikit-image',
                      'pyyaml'],
    entry_points={
        'console_scripts': ['dicomml=dicomml.tasks.main:main']},
    # Details
    url="https://github.com/volkherscholz/dicomml",
    license="LICENSE",
    description="Machine Learning with medical images",
    long_description=open("README.md").read())

import setuptools
from smpl._version import __version__

with open("README.rst", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setuptools.setup(
    name="smpl",
    version=__version__,
    author="SMPL",
    author_email="simpleenvironment@gmail.com",
    description="Process Control environments using gym API protocols",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smpl-env/smpl.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'': ['smpl/configdata*']},
    python_requires='>=3.8',
)

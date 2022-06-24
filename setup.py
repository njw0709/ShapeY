import setuptools

# reading long description from file
with open("README.md") as file:
    long_description = file.read()


# specify requirements of your package here
REQUIREMENTS = [
    "torch >= 1.10.0",
    "torchvision >= 0.11.2",
    "numpy>=1.18.4",
    "kaleido>=0.0.1",
    "tqdm>=4.48.0",
    "cupy-cuda102",
    "matplotlib>=3.3.1",
    "gdown>=4.0.2",
    "h5py>=2.10.0",
]

# some more details
CLASSIFIERS = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.6",
    "Development Status :: 4 - Beta",
]

# calling the setup function
setuptools.setup(
    name="ShapeY",
    version="0.0.16",
    description="Benchmark that tests shape recognition",
    long_description=long_description,
    url="https://github.com/njw0709/ShapeY",
    author="Jong Woo Nam",
    author_email="namj@usc.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords="tests shape recognition capacity",
)

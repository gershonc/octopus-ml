from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="octopus-ml",
    version="3.0.0",
    description="A collection of handy ML and data validation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["octopus_ml"],
    author="Gershon Celniker",
    author_email="gershonc@gmail.com",
    url="https://github.com/gershonc/octopus-ml",
    license="MIT",
    install_requires=["numpy", "pandas", "tqdm", "lightgbm", "seaborn"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
    ],
)

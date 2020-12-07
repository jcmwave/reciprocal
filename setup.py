from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="reciprocal", # Replace with your own username
    version="0.0.0-alpha.1",
    author="Phillip Manley",
    author_email="phillip.manley@helmholtz-berlin.de",
    description="utility functions for sampling reciprocal space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'matplotlib', 'scipy'],
    python_requires='>=3.6',
    include_package_data=True,
)

#data_files=[('config',['cfg/config.yaml'])],

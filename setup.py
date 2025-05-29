from setuptools import find_namespace_packages, setup

setup(
    name="campeones_analysis",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "mne",
        "mne-bids",
        "numpy",
        "pandas",
        "matplotlib",
        "pyxdf",
        "mnelab",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Analysis tools for the Campeones EEG dataset",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/campeones_analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

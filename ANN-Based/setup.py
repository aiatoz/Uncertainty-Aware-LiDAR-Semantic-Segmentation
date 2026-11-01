from setuptools import setup, find_packages

setup(
    name="uncertainty_pc",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*", "data", "data.*"]),
    python_requires=">=3.9",
)

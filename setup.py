"""Setup script for Taxi Benchmark Framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="taxi-benchmark",
    version="1.0.0",
    author="Taxi Benchmark Team",
    description="AWS Lambda-based framework for ride-hailing pricing experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/taxi-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "taxi-benchmark=taxi_benchmark_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "taxi_benchmark": ["config/*.yaml", "notebooks/*.ipynb"],
    },
) 
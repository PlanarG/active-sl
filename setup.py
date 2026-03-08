from setuptools import setup, find_packages

setup(
    name="benchmark",
    version="0.1.0",
    packages=find_packages(),
    package_data={"benchmark": ["dataset/**/*.parquet"]},
    install_requires=[
        "numpy",
        "scipy",
        "pyarrow",
        "matplotlib",
    ],
    python_requires=">=3.9",
)
from pathlib import Path
from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(
    encoding="utf-8")

setup(
    name="mfcs",
    version="0.1.0",
    description=("Multi-frequency component separation and large-scale "
                 "tomography for cross-correlations between large-scale "
                 "structure tracers and sky maps"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amy Wayland",
    url="https://github.com/amy-wayland/MF-component-separation",
    packages=find_packages(include=["mfcs", "mfcs.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0",  # np.trapezoid
        "scipy",
        "sacc",
        "pyccl>=3.0",
        "emcee",
    ],
    extras_require={
        # Additional packages needed to run the example notebooks
        "examples": ["healpy", "matplotlib", "jupyter"],
    },
)

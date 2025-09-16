from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        "networkx>=3.0",
        "numpy>=2.0.0",
        "scipy>=1.13.0",
        "matplotlib>=3.8.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.4.0",
        "shapely>=2.0.0",
    ],
    python_requires=">=3.8",
)
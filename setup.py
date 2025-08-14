from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        "networkx>=3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
)
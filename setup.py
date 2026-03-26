from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="spectrascope",
    version="1.0.0",
    description="Interpretable Machine learning toolkit for spectral data: feature generation, model training, and visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Toyota Research Institute",
    url="https://github.com/TRI-AMDD/spectrascope",
    license="Apache-2.0",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas",
        "pymatgen==2024.4.13",
        "scikit-learn==1.5.0",
        "PyWavelets==1.5.0",
        "scipy==1.13.1",
        "pyts==0.13.0",
        "streamlit==1.36.0",
        "plotly==5.22.0",
        "pdoc3==0.11.1",
        "matminer==0.9.2",
        "pillow==10.3.0",
        "matplotlib==3.8.3",
        "setuptools",
        "python-dotenv",
        "ace",
        "tqdm",
        "cvxpy",
    ],
    extras_require={
        "dev": ["pytest", "pre-commit", "black"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

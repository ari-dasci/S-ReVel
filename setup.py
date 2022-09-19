import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ReVel", # Replace with your own username
    version="0.0.1",
    author="Iván Sevillano Garcia",
    author_email="isevillano@ugr.es",
    description="Robust Evaluation VEctorized Local-linear-explanation",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    packages=["ReVel"],
    install_requires=['torch',
                      'scikit-learn',
                      'scikit-image',
                      'matplotlib',
                      'numpy',
                      'scipy',
                      'pandas',
                      'torchvision',
                      'efficientnet_pytorch',
                      'tqdm',
                      'opencv-python',
                      'IPython',
                      'seaborn',
                      'plotly',
                      'ipywidgets',
                      'sphinx',
                      'sphinx_rtd_theme',
                      'sphinxcontrib.bibtex',
                      'nbsphinx',
                      'wget'],
)
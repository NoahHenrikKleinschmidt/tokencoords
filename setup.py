from setuptools import setup, find_packages

setup(
    name="tokencoords",
    version="0.1.0",
    author="Noah Kleinschmidt",
    author_email="noah.kleinschmidt@unibe.ch",
    description="Tokenize coordinates for image processing in VLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NoahHenrikKleinschmidt/tokencoords",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Add your project dependencies here
        "numpy>=1.18.0",
    ],
)

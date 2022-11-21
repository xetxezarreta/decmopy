import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decmopy",
    version="1.0.0",
    author="Xabier Etxezarreta",
    description="Python implementation of DECMO algorithms inside the JMetalPy framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xetxezarreta/decmopy",
    packages=setuptools.find_packages(),
    install_requires=["jmetalpy==1.5.5"],
)

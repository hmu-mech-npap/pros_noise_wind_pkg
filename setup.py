import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pros_noisefiltering", # Replace with your own username
    version="0.0.5",
    author="N. Papadakis, N. Torosian",
    author_email="npapnet@gmail.com, goodvibrations32@protonmail.com",
    description="A package for processing data from Wind Turbine Blades at Wind Energ Laboratory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="argtyped",
    version="0.1",
    url="https://github.com/huzecong/argtyped",

    author="Zecong Hu",
    author_email="huzecong@gmail.com",

    description="Command line arguments, with types",
    long_description=long_description,
    long_description_content_type="text/markdown",

    license="MIT License",

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: System :: Shells",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
    python_requires='>=3.6',
)

import setuptools

setuptools.setup(
    name="snemo_gen",
    version="0.0.1",
    author="Sam Dixon",
    author_email="samdixon526@gmail.com",
    description="Kernel density estimates of SALT and SNEMO parameter distributions",
    url="https://github.com/sam-dixon/embed2spec",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
)
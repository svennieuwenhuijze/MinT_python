from setuptools import setup, find_packages


# Package meta-data.
NAME = 'Multivariate_fc'
DESCRIPTION = 'multivariate forecast.'
URL = '...'
EMAIL = '...'
AUTHOR = '...'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = (0, 1, 0)

#Where the magic happens:
setup(
    name=NAME,
    version='.'.join(map(str, VERSION)),
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages= find_packages(),
    include_package_data=True,
)

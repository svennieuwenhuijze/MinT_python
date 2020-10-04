# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:48:57 2020

@author: svenn
"""

from setuptools import setup, find_packages


# Package meta-data.
NAME = 'Multivariate_fc'
DESCRIPTION = 'Honeycomb multivariate forecast.'
URL = 'https://bitbucket.org/eyeon/multi_fc'
EMAIL = 'info@eyeon.nl'
AUTHOR = 'EyeOn'
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

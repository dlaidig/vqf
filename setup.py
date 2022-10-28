# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# https://stackoverflow.com/a/60740179
# (note that even with pyproject.toml this is still useful to make `python setup.py sdist` work out-of-the-box)
from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

import site
import sys
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

# workaround for develop mode (pip install -e) with PEP517/pyproject.toml cf. https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = '--user' in sys.argv[1:]

ext_modules = cythonize([
    'vqf/vqf.pyx',
    'vqf/basicvqf.pyx',
])

for m in ext_modules:
    m.include_dirs.insert(0, np.get_include())

setup(
    name='vqf',
    version='2.0.0',

    description='A Versatile Quaternion-based Filter for IMU Orientation Estimation',
    long_description=open('README.rst').read(),
    long_description_content_type="text/x-rst",
    url='https://github.com/dlaidig/vqf/',
    project_urls={
        'Documentation': 'https://vqf.readthedocs.io/',
    },

    author='Daniel Laidig',
    author_email='laidig@control.tu-berlin.de',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_data={'vqf': ['matlab/VQF.m', 'cpp/*.cpp', 'cpp/*.hpp', 'cpp/CMakeLists.txt']},
    zip_safe=False,

    install_requires=['numpy >= 1.20.0'],
    python_requires='>=3.7',  # needed for dataclasses in PyVQF
    extras_require={
        # pip3 install --user -e ".[dev]"
        'dev': ['tox', 'pytest', 'flake8',
                'reuse', 'transplant', 'oct2py', 'scipy', 'breathe', 'matplotlib',
                'sphinx', 'sphinx-rtd-theme', 'sphinxcontrib-matlabdomain'],
    },
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)

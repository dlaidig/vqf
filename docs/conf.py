# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# cf. https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

# always run doxygen automatically and abort if there is an error
subprocess.check_call('cd .. && mkdir -p build/doxygen && doxygen', shell=True)

project = 'VQF'
copyright = '2021, Daniel Laidig'
author = 'Daniel Laidig'

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'breathe',
    'sphinxcontrib.matlab',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
breathe_projects = {'vqf': '../build/doxygen/xml/'}
breathe_default_project = 'vqf'
matlab_src_dir = os.path.abspath('../vqf/')
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

autoclass_content = 'both'
autodoc_member_order = 'bysource'

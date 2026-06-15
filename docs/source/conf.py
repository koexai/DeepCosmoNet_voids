# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../')) # Percorso alla root del tuo progetto

autodoc_mock_imports = [m.replace(".py","") for m in os.listdir(os.path.abspath('../../src'))]

autodoc_mock_imports += ['GradScaler','psutil', 'torch.amp']

print(autodoc_mock_imports)


project = 'DeepCosmoNet Voids'
copyright = '2026, Vincenzo Del Zoppo'
author = 'Vincenzo Del Zoppo'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon' # Opzionale: per docstring in stile Google/NumPy
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

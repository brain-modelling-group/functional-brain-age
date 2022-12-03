#!/usr/bin/env python

'''
Functional brain age (FBA) example, basic setup to pull necessary libraries
with pip (pr preferrably pip3) install
'''

from setuptools import setup, find_packages
import os
import runpy

# Get the current folder
cwd = os.path.abspath(os.path.dirname(__file__))

#Minimal functionality
requirements = [
        'matplotlib',   # Plotting
        'numpy',        # Numerical functions
        'scipy',        # Scientific python and signal processing
        'onnxruntime']  # Performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.

# Get version
versionpath = os.path.join(cwd, 'version.py')
version = runpy.run_path(versionpath)['__version__']

setup(
    name='fba',
    version=version,
    author='Nathan Stevenson',
    author_email='pmsl.academic@gmail.com',         # TODO: for testing purposes, to change 
    description='Functional Age Brain predictions',
    url='http://github.com/pausz/fba-example',      # TODO: update url
    keywords=['scientific', 'ai', 'prediction'],
    platforms=['OS Independent'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)

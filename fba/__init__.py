#!/usr/bin/env python
"""
Functional Brain Age root module
This init is here so we can access the
functions in core.py without having to do

either:
import fba.core

or:
import fba.core as fba

We simply do:
import fba

or

or (less preferred):
from fba import *

"""

from .core import *

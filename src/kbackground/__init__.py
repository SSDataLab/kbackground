#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import  # noqa

import logging  # noqa
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
__version__ = "0.1.7"

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


from .kbackground import Estimator  # noqa

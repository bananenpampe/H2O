import os
import unittest

#add to syspath the ../utils/load.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model","utils"))

import dataset
import ase
import numpy as np
import rascaline
import torch
import metatensor


class TestJoin(unittest.TestCase):
    
        def test_join(self):
            """
            assert that the join function returns a dataset object
            """




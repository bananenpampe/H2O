import os
import unittest

#add to syspath the ../utils/load.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))

import dataset
import ase.io
import numpy as np
import rascaline
import torch
import equistore

import rascaline_torch
import rascaline
from nn.linear import EquistoreLinearLazy


PATH_TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "xtb_frames_for_testing.xyz")

hypers_sr = {
    "cutoff": 3.0,
    "max_radial": 5,
    "max_angular": 3,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 3.0, "rate": 1.5, "scale": 2.0}}
}



torch.set_default_dtype(d=torch.float64)



class TestEquistore(unittest.TestCase):

    def test_forward_pass(self):

        frames = ase.io.read(PATH_TEST_DATA, index=":")

        frames_t = [rascaline_torch.as_torch_system(frame) for frame in frames]
        calc = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))

        X = calc(frames_t)


        layer = EquistoreLinearLazy(n_out=10)
        layer.initialize_weights(X)
        X = layer(X)

        # now the blocks in X have different dimensionality
        # (they all should have 10 feat)

        layer = EquistoreLinearLazy(n_out=10)
        layer.initialize_weights(X)
        X = layer(X)

        # now, since the input shape is the same as the output shape
        # this should work:

        X = layer(X)
    

    def test_forward_pass_fail(self): 
        frames = ase.io.read(PATH_TEST_DATA, index=":")

        frames_t = [rascaline_torch.as_torch_system(frame) for frame in frames]
        calc = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))

        X = calc(frames_t)
        layer = EquistoreLinearLazy(n_out=10)
        layer.initialize_weights(X)
        X = layer(X)

        # this should fail, as the input dimensionality is changed to 10
        with self.assertRaises(RuntimeError):
            X = layer(X)
    
    def test_forward_pass_more_samples_forward(self):
        frames = ase.io.read(PATH_TEST_DATA, index=":")

        frames_t = [rascaline_torch.as_torch_system(frame) for frame in frames]
        calc = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))

        X = calc(frames_t)
        layer = EquistoreLinearLazy(n_out=10)
        layer.initialize_weights(X)

        X = equistore.join([X,X],axis="samples")
        X = layer(X)

    def test_forward_pass_different_blocks(self):
        frames = ase.io.read(PATH_TEST_DATA, index=":")

        frames_t = [rascaline_torch.as_torch_system(frame) for frame in frames]
        calc = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))

        X = calc(frames_t)

        layer = EquistoreLinearLazy(n_out=10)
        layer.initialize_weights(X)

        for frame in frames_t:
            X_int = calc(frames_t)
            X_int = layer(X_int)
    
    def test_w_components(self):
        frames = ase.io.read(PATH_TEST_DATA, index=":")

        frames_t = [rascaline_torch.as_torch_system(frame) for frame in frames]
        calc = rascaline_torch.Calculator(rascaline.SphericalExpansion(**hypers_sr))

        X = calc(frames_t)

        with self.assertRaises(AssertionError):
            layer = EquistoreLinearLazy(n_out=10)
            layer.initialize_weights(X)


if __name__ == '__main__':
    unittest.main()


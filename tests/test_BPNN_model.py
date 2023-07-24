
import os
import unittest

#add to syspath the ../utils/load.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))


#from utils import load
import dataset
import ase.io
import numpy as np
import rascaline
import torch
import equistore


import rascaline_torch
import rascaline
from nn.linear import EquistoreLinearLazy
from nn.model import BPNNModel
from nn.interaction import BPNNInteraction
from nn.feature import UnitFeatures
from nn.aggregation import BPNNStructureWiseAggregation
from nn.response import ForceRespone
from nn.nonlinear import EquistoreMLPLazy
from nn.soap import compute_power_spectrum


torch.set_default_dtype(d=torch.float64)
# make preparation tests.
# test that model can be properly initialized

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

hypers_lr = dict( cutoff = 3.0,
max_radial = 6,
max_angular = 2,
atomic_gaussian_width = 1.0,
center_atom_weight = 1.0,
radial_basis = dict( Gto = {} ),
potential_exponent = 1
)


class TestProperInit(unittest.TestCase):

    def test_model_init(self):
        """
        assert that the model can be properly initialized

        and API has not changed
        """
        model = BPNNModel()

        assert isinstance(model.feature, UnitFeatures)
        assert isinstance(model.interaction, BPNNInteraction)
        assert isinstance(model.aggregation, BPNNStructureWiseAggregation)
        assert isinstance(model.response, ForceRespone)
        assert isinstance(model.interaction.model, EquistoreMLPLazy)
        
        #assert model.interaction.model.n_hidden_layers == 2
        #assert model.interaction.model.n_hidden == 32
        #assert model.interaction.model.n_out == 1
        #assert model.interaction.model.activation == torch.nn.Tanh

        assert model.aggregation.mode == "sum"
        assert model.aggregation.sum_over == ["species_center","center"]
        
    def test_model_init_w_map(self):

        frames = ase.io.read("test_H2O.xyz", index=":2")
        frames = [rascaline_torch.as_torch_system(frame) for frame in frames]

        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))
        X = calculator(frames)

        model = BPNNModel()
        model.initialize_weights(X)

        assert isinstance(model.feature, UnitFeatures)
        assert isinstance(model.interaction, BPNNInteraction)
        assert isinstance(model.aggregation, BPNNStructureWiseAggregation)
        assert isinstance(model.response, ForceRespone)
        assert isinstance(model.interaction.model, EquistoreMLPLazy)

        assert model.aggregation.mode == "sum"
        assert model.aggregation.sum_over == ["species_center","center"]

        assert len(model.interaction.model.m_map) == len(X)
        
        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))
        X = calculator(frames)

        model = BPNNModel()
        model.initialize_weights(X)


    def test_size_extensivity(self):
        """ Tests, wether the BPNN model is size extensive with model size
        """

        frames_ase = ase.io.read("test_H2O.xyz", index=":2")
        frames = [rascaline_torch.as_torch_system(frame,
                                                  positions_requires_grad=True)
                                                  for frame in frames_ase]

        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))
        X = calculator(frames)
        X = X.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        model = BPNNModel()
        model.initialize_weights(X)

        E = model.forward(X, frames)
        E = E.block(0).values

        frames_2 = [frame * (2,1,1) for frame in frames_ase]
        frames_3 = [frame * (3,1,1) for frame in frames_ase]

        frames_2 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames_2]
        
        frames_3 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True )
                                                    for frame in frames_3]

        X_2 = calculator(frames_2)
        X_2 = X_2.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_3 = calculator(frames_3)
        X_3 = X_3.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        E_2 = model.forward(X_2, frames_2)
        E_2 = E_2.block(0).values

        E_3 = model.forward(X_3, frames_3)
        E_3 = E_3.block(0).values


        #are the energies non-zero?
        self.assertTrue(torch.all(E.ne(0)))
        self.assertTrue(torch.all(E_2.ne(0)))
        self.assertTrue(torch.all(E_3.ne(0)))

        # are they size extensive?
        self.assertTrue(torch.allclose(E * 2., E_2))
        self.assertTrue(torch.allclose(E * 3., E_3))

        #TODO: How do we assert that it is not just the size of the input (ie the bias)

        # add noise to positions and check if energy changes:
    
    def skip_test_size_extensivity_w_spex(self):
        """ Tests, wether the BPNN model is size extensive with spherical expansions
        """

        frames_ase = ase.io.read("test_H2O.xyz", index=":2")
        frames = [rascaline_torch.as_torch_system(frame,
                                                  positions_requires_grad=True)
                                                  for frame in frames_ase]

        calculator = rascaline_torch.Calculator(rascaline.SphericalExpansion(**hypers_sr))

        X = calculator(frames)
        X = compute_power_spectrum(X, naming_convention="soap")
        X = X.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        model = BPNNModel()
        model.initialize_weights(X)

        E = model.forward(X, frames)
        E = E.block(0).values

        frames_2 = [frame * (2,1,1) for frame in frames_ase]
        frames_3 = [frame * (3,1,1) for frame in frames_ase]

        frames_2 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames_2]
        
        frames_3 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True )
                                                    for frame in frames_3]

        X_2 = calculator(frames_2)
        X_2 = compute_power_spectrum(X_2, naming_convention="soap")
        X_2 = X_2.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_3 = calculator(frames_3)
        X_3 = compute_power_spectrum(X_3, naming_convention="soap")
        X_3 = X_3.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        E_2 = model.forward(X_2, frames_2)
        E_2 = E_2.block(0).values

        E_3 = model.forward(X_3, frames_3)
        E_3 = E_3.block(0).values

        #are the energies non-zero?
        self.assertTrue(torch.all(E.ne(0)))
        self.assertTrue(torch.all(E_2.ne(0)))
        self.assertTrue(torch.all(E_3.ne(0)))

        # are they size extensive?
        self.assertTrue(torch.allclose(E * 2., E_2))
        self.assertTrue(torch.allclose(E * 3., E_3))

    def skip_test_size_extensivity_w_spex_join(self):
        """ Tests, wether the BPNN model is size extensive with spherical expansions
        and joining of descriptors
        """

        frames_ase = ase.io.read("test_H2O.xyz", index=":2")
        frames = [rascaline_torch.as_torch_system(frame,
                                                  positions_requires_grad=True)
                                                  for frame in frames_ase]

        calculator = rascaline_torch.Calculator(rascaline.SphericalExpansion(**hypers_sr))

        X = calculator(frames)
        X = compute_power_spectrum(X, naming_convention="soap")
        X = X.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X = equistore.join([X, X], axis="properties")

        model = BPNNModel()
        model.initialize_weights(X)

        E = model.forward(X, frames)
        E = E.block(0).values

        frames_2 = [frame * (2,1,1) for frame in frames_ase]
        frames_3 = [frame * (3,1,1) for frame in frames_ase]

        frames_2 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames_2]
        
        frames_3 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True )
                                                    for frame in frames_3]

        X_2 = calculator(frames_2)
        X_2 = compute_power_spectrum(X_2, naming_convention="soap")
        X_2 = X_2.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_2 = equistore.join([X_2, X_2], axis="properties")

        X_3 = calculator(frames_3)
        X_3 = compute_power_spectrum(X_3, naming_convention="soap")
        X_3 = X_3.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_3 = equistore.join([X_3, X_3], axis="properties")

        E_2 = model.forward(X_2, frames_2)
        E_2 = E_2.block(0).values

        E_3 = model.forward(X_3, frames_3)
        E_3 = E_3.block(0).values

        #are the energies non-zero?
        self.assertTrue(torch.all(E.ne(0)))
        self.assertTrue(torch.all(E_2.ne(0)))
        self.assertTrue(torch.all(E_3.ne(0)))

        # are they size extensive?
        self.assertTrue(torch.allclose(E * 2., E_2))
        self.assertTrue(torch.allclose(E * 3., E_3))


    def skip_test_size_extensivity_sr_lr(self):
        print("begin")
        frames_ase = ase.io.read("test_H2O.xyz", index=":1")
        frames = [rascaline_torch.as_torch_system(frame,
                                                positions_requires_grad=True)
                                                for frame in frames_ase]

        calculator_sr = rascaline_torch.Calculator(rascaline.SphericalExpansion(**hypers_sr))
        calculator_lr = rascaline_torch.Calculator(rascaline.LodeSphericalExpansion(**hypers_lr))


        X_sr = calculator_sr(frames)
        X_lr = calculator_lr(frames)

        X_sr = compute_power_spectrum(X_sr, naming_convention="soap")
        X_sr = X_sr.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_lr = compute_power_spectrum(X_lr, naming_convention="soap")
        X_lr = X_lr.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X = equistore.join([X_sr, X_lr], axis="properties")
        
        model = BPNNModel()
        model.initialize_weights(X)

        E = model.forward(X, frames)
        E = E.block(0).values

        frames_2 = [frame * (2,1,1) for frame in frames_ase]
        frames_3 = [frame * (3,1,1) for frame in frames_ase]

        frames_2 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames_2]
        
        frames_3 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True )
                                                    for frame in frames_3]

        X_sr_2 = calculator_sr(frames_2)
        X_lr_2 = calculator_lr(frames_2)

        X_sr_2 = compute_power_spectrum(X_sr_2, naming_convention="soap")

        X_lr_2 = compute_power_spectrum(X_lr_2, naming_convention="soap")

        X_2 = equistore.join([X_sr_2, X_lr_2], axis="properties")
        X_2 = X_2.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_sr_3 = calculator_sr(frames_3)
        X_lr_3 = calculator_lr(frames_3)

        X_sr_3 = compute_power_spectrum(X_sr_3, naming_convention="soap")
        X_lr_3 = compute_power_spectrum(X_lr_3, naming_convention="soap")

        X_3 = equistore.join([X_sr_3, X_lr_3], axis="properties")
        X_3 = X_3.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        E_2 = model.forward(X_2, frames_2)
        E_2 = E_2.block(0).values

        E_3 = model.forward(X_3, frames_3)
        E_3 = E_3.block(0).values

        self.assertTrue(torch.allclose(E * 2., E_2))
        self.assertTrue(torch.allclose(E * 3., E_3))

    def skip_test_equal_featslr(self):
        
        frames_ase = ase.io.read("test_H2O.xyz", index=":1")
        frames = [rascaline_torch.as_torch_system(frame,
                                            positions_requires_grad=True)
                                            for frame in frames_ase]

        frames_2 = [frame * (2,1,1) for frame in frames_ase]
        frames_2 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames_2]
        
        calculator_lr = rascaline_torch.Calculator(rascaline.LodeSphericalExpansion(**hypers_lr))
        
        X_lr = calculator_lr(frames)
        X_lr_2 = calculator_lr(frames_2)

        X_lr = compute_power_spectrum(X_lr, naming_convention="soap")
        X_lr = X_lr.keys_to_samples(["species_center"])
        X_lr = X_lr.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_lr_2 = compute_power_spectrum(X_lr_2, naming_convention="soap")
        X_lr_2 = X_lr_2.keys_to_samples(["species_center"])
        X_lr_2 = X_lr_2.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        s_lr_2 = X_lr_2.block(0).values.shape[0] // 2

        self.assertTrue(torch.allclose(X_lr.block(0).values, X_lr_2.block(0).values[:s_lr_2,:]))


        

    def skip_test_rascaline_calc(self):
        frames_ase = ase.io.read("test_H2O.xyz", index=":2")
        frames = [rascaline_torch.as_torch_system(frame,
                                                  positions_requires_grad=True)
                                                  for frame in frames_ase]

        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))
        X = calculator(frames)
        X = X.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        model = BPNNModel()
        model.initialize_weights(X)

        E = model.forward(X, frames)
        E = E.block(0).values

        frames_2 = [frame.r for frame in frames_ase]

        frames_2 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames_2]
        
        frames_3 = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True )
                                                    for frame in frames_3]

        X_2 = calculator(frames_2)
        X_2 = X_2.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        X_3 = calculator(frames_3)
        X_3 = X_3.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        E_2 = model.forward(X_2, frames_2)
        E_2 = E_2.block(0).values

        E_3 = model.forward(X_3, frames_3)
        E_3 = E_3.block(0).values

        self.assertTrue(torch.allclose(E * 8., E_2))
        self.assertTrue(torch.allclose(E * 27., E_3))
        pass


    def test_rotational_invariance(self):

        frames = ase.io.read("test_H2O.xyz", index=":1")
        frames = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames]
        
        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))
        X = calculator(frames)
        X = X.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        model = BPNNModel()
        model.initialize_weights(X)
        
        E = model.forward(X, frames)
        E = E.block(0).values

        for i in range(20):
            pass

    def test_translational_invariance(self):

        frames = ase.io.read("test_H2O.xyz", index=":1")
        frames = [rascaline_torch.as_torch_system(frame,
                                                    positions_requires_grad=True)
                                                    for frame in frames]
        
        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**hypers_sr))
        X = calculator(frames)
        X = X.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

        model = BPNNModel()
        model.initialize_weights(X)
        
        E = model.forward(X, frames)
        E = E.block(0).values

        #make non zero and rattle tests
        # call -> rattle(0.1)

        for i in range(20):

            ax = np.random.choice(["x","-x","y","-y","z","-z"])
            angle = np.random.uniform(0,360)
            frame_test = frames[0].copy()
            frame_test.rotate(angle, ax)

            frame_test = rascaline_torch.as_torch_system(frame_test,
                                                    positions_requires_grad=True)
            
            E_test = model.forward(X, [frame_test])





    def test_atom_listing_invariance(self):


        #shuffle ids
        # and then set correct forces, and positions

        frames = ase.io.read("test_H2O.xyz", index=":2")

        for frame in frames:
            
            for i in range(10): 

                frame_copy = frame.copy()
                
                z = frame.get_atomic_numbers()
                idx = np.arange(len(z))
                np.random.shuffle(idx)
                frame_copy.set_atomic_numbers(z[idx])
                frame_copy.set_positions(frame.get_positions()[idx])
                frame_copy.set_cell(frame.get_cell())
                frame_copy.set_pbc(frame.get_pbc())




        pass
    
    def test_forces_finite_differences(self):
        pass















# 1st test. A model should yield exactly twice the energy if the unit cell is doubled
# 1.1 check that this applies for forces aswell

# 2nd test. A model should yield exactly the same energy if the unit cell is rotated
# now the forces should be rotated aswell

# 3rd test. A model should yield exactly the same energy if the unit cell is translated
# 3.1 check that forces are identical

# 4th test. A model should yield exactly the same energy if the unit cell is rotated and translated

# 5th test. A model should yield exactly the same energy if the unit cell is rotated and translated and the atoms are shuffled

# 6th test. A model should yield exactly the twice energy if the unit cell is rotated and translated and the atoms are shuffled and the cell is doubled

# 7th show consisentcy for multiplication of n*times the unit cell

# 8th show consisentcy for multiplication of n*times the unit cell and rotation

# 9th show that the model is invariant to the order of the atoms




#ALL TESTS: make sure that the model outputs something else than zero:
# (and something else than totally saturated)
# 1.1 check that this applies for forces aswell
# check that it is doing something else than just 



# Make a finite difference test for the forces:
# take one force component (N_samples=1, x) and check that it is equal to E(x+dx) - E(x-dx) / 2dx



if __name__ == '__main__':
    unittest.main()



# Testcase: test_transformer.py

import os
import unittest

#add to syspath the ../utils/load.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))


from load import load_PBE0_TS
import dataset
import ase.io
import numpy as np
import rascaline
import torch
import metatensor

import rascaline_torch
import rascaline
from nn.linear import metatensorLinearLazy
from equisolve_futures.convert import ase_to_tensormap

import rascaline_torch

torch.set_default_dtype(d=torch.float64)


from transformer.composition import CompositionTransformer

frames = ase.io.read("test_H2O.xyz", index=":")

targets = ase_to_tensormap(frames,"energy","forces","stress")
test_frames = [rascaline_torch.as_torch_system(frame) for frame in frames]
targets = metatensor.to(targets, "torch")

mean_energy = torch.mean(targets.block(0).values)
std_energy = torch.std(targets.block(0).values)


# do a stupid second case, where energies 


class TestCompositionTransformer(unittest.TestCase):

    def test_energy(self):
        """
        assert that the energy is the same before and after the transformation
        """
        
        transformer = CompositionTransformer()
        transformer.fit(test_frames,targets)
        transformed_targets = transformer.forward(test_frames,targets)
        back_transformed_targets = transformer.inverse_transform(test_frames,transformed_targets)

        print("mean energy before transformation: ", mean_energy)
        print("mean energy after transformation: ", torch.mean(transformed_targets.block(0).values))
        print("mean energy after back transformation: ", torch.mean(back_transformed_targets.block(0).values))

        print("std energy before transformation: ", std_energy)
        print("std energy after transformation: ", torch.std(transformed_targets.block(0).values))
        print("std energy after back transformation: ", torch.std(back_transformed_targets.block(0).values))

    def test_size_consitency_should_break(self):
        """
        asserts that a transformer, eventhough trained on fixed system size can transform for larger system sizes
        """

        frames_2 = [frame*2 for frame in frames] # multiplied 8 fold (2x2x2)
        
        for frame in frames_2:
            frame.info["energy"] *= 8
        
        targets_train = ase_to_tensormap(frames,"energy","forces","stress")
        targets_test = ase_to_tensormap(frames_2,"energy","forces","stress")
        
        fit_frames = [rascaline_torch.as_torch_system(frame) for frame in frames]
        test_frames = [rascaline_torch.as_torch_system(frame) for frame in frames_2]

        targets_train = metatensor.to(targets_train, "torch")
        targets_test = metatensor.to(targets_test, "torch")

        mean_energy_train = torch.mean(targets_train.block(0).values)
        std_energy_train = torch.std(targets_train.block(0).values)
        
        mean_energy_test = torch.mean(targets_test.block(0).values)
        std_energy_test = torch.std(targets_test.block(0).values)

        transformer = CompositionTransformer()
        transformer.fit(fit_frames,targets)
        
        transformed_targets = transformer.forward(fit_frames,targets_train)
        back_transformed_targets = transformer.inverse_transform(fit_frames,transformed_targets)

        print("train mean energy before transformation: ", mean_energy_train)
        print("train mean energy after transformation: ", torch.mean(transformed_targets.block(0).values))
        print("train mean energy after back transformation: ", torch.mean(back_transformed_targets.block(0).values))

        print("train std energy before transformation: ", std_energy_train)
        print("train std energy after transformation: ", torch.std(transformed_targets.block(0).values))
        print("train std energy after back transformation: ", torch.std(back_transformed_targets.block(0).values))

        transformed_targets = transformer.forward(test_frames,targets_test)
        back_transformed_targets = transformer.inverse_transform(test_frames,transformed_targets)

        print("test mean energy before transformation: ", mean_energy_test)
        print("test mean energy after transformation: ", torch.mean(transformed_targets.block(0).values))
        print("test mean energy after back transformation: ", torch.mean(back_transformed_targets.block(0).values))

        print("test std energy before transformation: ", std_energy_test)
        print("test std energy after transformation: ", torch.std(transformed_targets.block(0).values))
        print("test std energy after back transformation: ", torch.std(back_transformed_targets.block(0).values))


    def test_size_consitency_should_not_break(self):
        """
        asserts that a transformer, eventhough trained on fixed system size can transform for larger system sizes
        """

        frames_2 = [frame*2 for frame in frames] # multiplied 8 fold (2x2x2)
        
        for frame in frames_2:
            frame.info["energy"] *= 8

        targets_train = ase_to_tensormap(frames,"energy","forces","stress")
        targets_test = ase_to_tensormap(frames_2,"energy","forces","stress")
        
        fit_frames = [rascaline_torch.as_torch_system(frame) for frame in frames]
        test_frames = [rascaline_torch.as_torch_system(frame) for frame in frames_2]

        targets_train = metatensor.to(targets_train, "torch")
        targets_test = metatensor.to(targets_test, "torch")

        mean_energy_train = torch.mean(targets_train.block(0).values)
        std_energy_train = torch.std(targets_train.block(0).values)
        
        mean_energy_test = torch.mean(targets_test.block(0).values)
        std_energy_test = torch.std(targets_test.block(0).values)

        transformer = CompositionTransformer(bias=False)
        transformer.fit(fit_frames,targets)
        
        transformed_targets = transformer.forward(fit_frames,targets_train)
        back_transformed_targets = transformer.inverse_transform(fit_frames,transformed_targets)

        print("train mean energy before transformation: ", mean_energy_train)
        print("train mean energy after transformation: ", torch.mean(transformed_targets.block(0).values))
        print("train mean energy after back transformation: ", torch.mean(back_transformed_targets.block(0).values))

        print("train std energy before transformation: ", std_energy_train)
        print("train std energy after transformation: ", torch.std(transformed_targets.block(0).values))
        print("train std energy after back transformation: ", torch.std(back_transformed_targets.block(0).values))

        transformed_targets = transformer.forward(test_frames,targets_test)
        back_transformed_targets = transformer.inverse_transform(test_frames,transformed_targets)

        print("test mean energy before transformation: ", mean_energy_test)
        print("test mean energy after transformation: ", torch.mean(transformed_targets.block(0).values))
        print("test mean energy after back transformation: ", torch.mean(back_transformed_targets.block(0).values))

        print("test std energy before transformation: ", std_energy_test)
        print("test std energy after transformation: ", torch.std(transformed_targets.block(0).values))
        print("test std energy after back transformation: ", torch.std(back_transformed_targets.block(0).values))





if __name__ == '__main__':
    unittest.main()

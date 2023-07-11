# write a unit test that checks that the function load_from_AISquare() works as expected.
#

import os
import unittest

#add to syspath the ../utils/load.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))
import load
import ase
import numpy as np

class TestLoadPBE0(unittest.TestCase):
    #first assert that the PBE0TS directory exists
    
    def test_PBE0TS(self):
        """
        assert that the data directory contains a PBE0-TS-H2O directory
        """
        print(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-PBE0TS"))
        assert os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-PBE0TS"))
    #then assert that the function load_PBE0_TS() works as expected
    
    def test_load_PBE0_TS(self):
        """
        assert that the load function returns ase atoms objects for each of the 4 systems
        """
        
        for option in ["lw_pimd","ice_triple_I","ice_triple_II","ice_pimd"]:
            atoms = load.load_PBE0_TS(option)
            self.assertTrue(isinstance(atoms[3], ase.Atoms))

    def test_type_map(self):
        """
        asserts that the type map is correct by checking that the ratio of hydrogen to oxygen in one frame is 2:1
        """

        atoms = load.load_PBE0_TS("lw_pimd")
        species = np.unique(atoms[5].get_atomic_numbers(), return_counts=True)

        self.assertEqual(species[0][0], 1)
        self.assertEqual(species[0][1], 8)
        self.assertEqual(species[1][0]//species[1][1], 2)
        self.assertEqual(len(species[1]), 2)

class TestLoadPhaseDiagram(unittest.TestCase):

    def test_PhaseDiagram(self):
        """
        assert that the data directory contains a H2O-Phase-Diagram directory
        """
        assert os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-Phase-Diagram"))

    def test_load_phase_diagram_H2O(self):
        """
        assert that the load function returns ase atoms objects for each of the 4 systems
        """

        atoms = load.load_phase_diagram_H2O()
        self.assertTrue(isinstance(atoms[3], ase.Atoms))

    def test_type_map(self):
        """
        asserts that the type map is correct by checking that the ratio of hydrogen to oxygen in one frame is 2:1
        """

        atoms = load.load_phase_diagram_H2O()
        species = np.unique(atoms[5].get_atomic_numbers(), return_counts=True)

        self.assertEqual(species[0][0], 1)
        self.assertEqual(species[0][1], 8)
        self.assertEqual(species[1][0]//species[1][1], 2)
        self.assertEqual(len(species[1]), 2)
    
    def test_len_frames(self):
        """
        asserts that the length of the frames is correct
        """
        atoms = load.load_phase_diagram_H2O()
        self.assertEqual(len(atoms), 48419)

if __name__ == '__main__':
    unittest.main()
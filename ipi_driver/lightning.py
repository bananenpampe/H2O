
import sys
import os

LIGHNING_CALCULATOR_PATH = os.environ["MLIP_DIR"] 

sys.path.append(LIGHNING_CALCULATOR_PATH)

from .dummy import Dummy_driver

from ipi.utils.mathtools import det_ut3x3
from ipi.utils.units import unit_to_internal, unit_to_user

sys.path.append(LIGHNING_CALCULATOR_PATH)

__DRIVER_NAME__ = "lightning"
__DRIVER_CLASS__ = "Lightning_driver"

from ipi_calculator_committee import PytorchLightningCalculator

class Lightning_driver(Dummy_driver):
    def __init__(self, args=None, verbose=None):
        print(args)
        self.error_msg = """Rascal driver requires specification of a .json model file fitted with librascal, 
                            and a template file that describes the chemical makeup of the structure. 
                            Example: python driver.py -m rascal -u -o example.chk,template.xyz"""

        super().__init__(args)

        if PytorchLightningCalculator is None:
            raise ImportError("Couldn't load librascal bindings")

    def check_arguments(self):
        """Check the arguments required to run the driver

        This loads the potential and atoms template in librascal
        """

        if len(self.args) == 2:
            self.model = self.args[0]
            self.template = self.args[1]
        else:
            sys.exit(self.error_msg)
        
        self.alchemical_calc = PytorchLightningCalculator(self.model, self.template)

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the librascal model"""
        pos_calc = unit_to_user("length", "angstrom", pos)
        # librascal expects ASE-format, cell-vectors-as-rows
        cell_calc = unit_to_user("length", "angstrom", cell.T)
        # Do the actual calculation
        
        pot, force, stress, extras = self.alchemical_calc.calculate(pos_calc, cell_calc)
        
        pot_ipi = unit_to_internal("energy", "electronvolt", pot)
        force_ipi = unit_to_internal("force", "ev/ang", force.reshape(-1, 3))

        # rascaline returns the stress in energy units already (i.e. as dV/deps)
        # TODO: implement actual virial calculation
        vir_calc = stress
        vir_ipi = unit_to_internal("energy", "electronvolt", vir_calc.T)
        #extras = ""
        
        return pot_ipi, force_ipi, vir_ipi, extras


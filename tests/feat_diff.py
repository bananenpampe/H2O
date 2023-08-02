import ase.io
import rascaline
import numpy as np

lode_hypers = dict( cutoff = 3.0,
max_radial = 6,
max_angular = 2,
atomic_gaussian_width = 1.0,
center_atom_weight = 1.0,
radial_basis = dict( Gto = {} ),
potential_exponent = 1
)

frames = ase.io.read("test_H2O_2.xyz", index=":2", format="extxyz")

# makes (2x1x1) supercells
frames_2 = [frame * (2,1,1) for frame in frames]

calc = rascaline.LodeSphericalExpansion(**lode_hypers)

feat = calc.compute(frames)
feat_2 = calc.compute(frames_2)

feat = feat.components_to_properties("spherical_harmonics_m")
feat = feat.keys_to_properties("spherical_harmonics_l")
feat = feat.keys_to_samples("species_center")
feat = feat.keys_to_properties(["species_neighbor"]) #,"species_neighbor_2"])

feat_2 = feat_2.components_to_properties("spherical_harmonics_m")
feat_2 = feat_2.keys_to_properties("spherical_harmonics_l")
feat_2 = feat_2.keys_to_samples("species_center")
feat_2 = feat_2.keys_to_properties(["species_neighbor"])


feat_slice = np.array(feat_2.block(0).values[:feat.block(0).values.shape[0]])

np.allclose(feat.block(0).values, feat_slice)
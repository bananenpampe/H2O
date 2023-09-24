# contains loading files for different water files

import dpdata
import os
import re
import os
import tqdm
import ase.io
import numpy as np
from ase import Atoms

def find_file_in_dirs(root_dir, filename):
    dirs_with_file = []

    # os.walk generates the file names in a directory tree by walking the tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # if the file exists in the current directory, add the directory to the list
        if filename in filenames:
            dirs_with_file.append(dirpath)
    return dirs_with_file



def load_PBE0_TS(which="lw_pimd"):

    """
    load the PBE0 TS data from the deepmd-npy format

    The options are:

    ice from pimd: ice_pimd
    water from pimd: lw_pimd

    ice from triple point I: ice_triple_I
    ice from triple point II: ice_triple_II
    """

    #assert that the data directory contains a H2O-PBE0TS directory
    assert os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-PBE0TS"))
    assert which in ["lw_pimd","ice_triple_I","ice_triple_II","ice_pimd"]

    #join the path of this file with the path of the data
    PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-PBE0TS", which)
    system = dpdata.LabeledSystem().from_deepmd_npy(PATH, type_map=["O","H"],fmt="deepmd/npy")

    return system.to_ase_structure()



def load_phase_diagram_H2O(load_ice_subset=False):
    """loads the entire H2O phase diagram dataset
    no distinction is made between different datasets
    """
    
    print("DO NOT USE, DATASET SIZE INCONSISTENT")

    #assert that the data directory contains a H2O-Phase-Diagram directory
    # 
    assert os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-Phase-Diagram"))

    PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "H2O-Phase-Diagram")

    filename = "type.raw"
    dirs = find_file_in_dirs(PATH, filename)

    all_systems = []
    identifiers = []

    for dir_ in dirs:
        #if "iter" in dir_:
            #print("in dir")
        #print(dir_)
        if load_ice_subset:
            matches = re.findall(r'(?<=/)([^/]*ice[^/]*)(?=/)', dir_)
            
            if len(matches) > 0:
                s = dpdata.LabeledSystem(dir_,type_map=["O","H"],fmt="deepmd/npy")
                all_systems.append(s)
                if "high_pressure" in dir_:
                    identifiers.append(matches[0]+"_hp")
                else:
                    identifiers.append(matches[0])
                print(matches[0])
        
        else:
            s = dpdata.LabeledSystem(dir_,type_map=["O","H"],fmt="deepmd/npy")
            all_systems.append(s)
        
    frames = []
    
    for n, s in enumerate(all_systems):
        
        if load_ice_subset:
            
            tmp_frames = s.to_ase_structure()
            
            for frame in tmp_frames:
                frame.info["identifier"] = identifiers[n]
            
            frames.extend(tmp_frames)
        
        else:
            frames.extend(s.to_ase_structure())

    return frames



# everything below is from: (Sergey Pozdnyakov)
# ------------- from: https://github.com/lab-cosmo/nice/blob/master/examples/qm9_small.ipynb !!! ------------


PROPERTIES_NAMES = ['tag', 'index', 'A', 'B', 'C', 'mu',
                    'alpha', 'homo', 'lumo', 'gap', 'r2',
                    'zpve', 'U0', 'U', 'H', 'G', 'Cv']


def string_to_float(element):
    '''because shit like 2.1997*^-6 happens'''
    return float(element.replace('*^', 'e'))

PROPERTIES_HANDLERS = [str, int] + [string_to_float] * (len(PROPERTIES_NAMES) - 2)

def parse_qm9_xyz(path):
    with open(path, 'r') as f:
        lines = list(f)
    #print(lines)

    #MODIFICATION TO ADD INCHI KEY
    inchi_ids = lines[-1].rstrip("\n").split("\t")

    assert len(inchi_ids) == 2

    n_atoms = int(lines[0])
    properties = {name:handler(value)
                  for handler, name, value in zip(PROPERTIES_HANDLERS,
                                            PROPERTIES_NAMES,
                                            lines[1].strip().split())}
    composition = ""
    positions = []
    for i in range(2, 2 + n_atoms):
        composition += lines[i].strip().split()[0]
        positions.append([string_to_float(value) 
                          for value in lines[i].strip().split()[1:4]])
        
    
    positions = np.array(positions)
    result = Atoms(composition, positions = np.array(positions))
    result.info.update(properties)
    result.info['inchi_key_0'] = inchi_ids[0]
    result.info['inchi_key_1'] = inchi_ids[1]

    return result

def parse_index(path):
    with open(path, "r") as f:
        lines = list(f)
    proper_lines = lines[9:-1]
    result = [int(line.strip().split()[0]) for line in proper_lines]
    return np.array(result, dtype = int)

def download_qm9(clean = True):
    #downloading from https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    os.system("wget https://ndownloader.figshare.com/files/3195389 -O qm9_main.xyz.tar.bz2")
    os.system("wget https://ndownloader.figshare.com/files/3195404 -O problematic_index.txt")
    os.system("mkdir qm9_main_structures")
    os.system("tar xjf qm9_main.xyz.tar.bz2 -C qm9_main_structures")
    
    names = [name for name in os.listdir('qm9_main_structures/') if name.endswith('.xyz')]
    names = sorted(names)
    
    structures = [parse_qm9_xyz('qm9_main_structures/{}'.format(name))
              for name in tqdm.tqdm(names)]
    
    problematic_index = parse_index('problematic_index.txt')
    np.save('problematic_index.npy', problematic_index)
    ase.io.write('qm9_main.extxyz', structures)
    if (clean):
        os.system("rm -r qm9_main_structures")
        os.system("rm problematic_index.txt")
        os.system("rm qm9_main.xyz.tar.bz2")
    return structures, problematic_index
              
def get_qm9(clean = True):
    if ('qm9_main.extxyz' in os.listdir('.')) and \
              ('problematic_index.npy' in os.listdir('.')):
        structures = ase.io.read('qm9_main.extxyz', index = ':')
        problematic_index = np.load('problematic_index.npy')
        return structures, problematic_index
    else:
        return download_qm9(clean = clean)
    

def get_qm9_w_problematic(clean=True):
    
    structures, problematic_index = get_qm9(clean=clean)
    
    for structure in structures:
        if structure.info['index'] in problematic_index:
            structure.info['problematic'] = "PROBLEMATIC"
        else:
            structure.info['problematic'] = "OK"
        
    return structures








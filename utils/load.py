# contains loading files for different water files

import dpdata
import os
import re

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









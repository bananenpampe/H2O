# design considerations for the dataset class


# takes ase Atoms Objects as input
# takes a series of calculators as input
# handle atom-wise / frame-wise properties ?

# properties/forces ?

# have a series of base-transformers, taht operate, ie on composition feat
# the transformer directly usses composition feats
# treat structure wise models, also as transformers? structure wise soap??

# compute all features in memory as features?
# have a compute-on-the-fly option for features?

# have a precompute option:
# if precompute is true:
# calculate all feats during instantation
# if precompute is false:
# calculate feats on the fly, in the getitem method
# in the getitem method, there should be a check if the feats are already calculated, 
# or if precompute is true

# have a precompute option?

# how to handle hyperparameters ?
# pass calculator instances instead of class names?

# have a save option:

# save only calculator hypers + frames
# save also features ?



# model ideas:
# have species wise blocks:
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "equisolve_futures"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "equisolve_futures"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "equistore_torch_operations_futures"))

import ase
import torch
from .dataset_helpers import get_global_unique_species
from convert_torch import ase_to_tensormap


from typing import List, Union, Tuple
import rascaline
import equistore
import equistore.torch
import rascaline.torch

import copy
from itertools import combinations_with_replacement
import numpy as np
from join import join

class RascalineAtomisticDataset(torch.utils.data.Dataset):
    """ A dataset for general rascaline calculators
    """

    # TODO: lazy-fill up
    # lazy fill up can be used when precompute is true
    # the features will be computed during the first epoch on the fly but stored in memory
    # if precompute is true, then the features are calculated during instantation


    def _compute_feats(self, frame, global_species):
        # currently we have the issue, that join functions do not tolerate diffent number of blocks.
        # if we have a system that has at least a constant global composition
        # and the features, are SOAP-like we can use this reshaping
        # TODO: at a leater stage we definetly need to change this
        
        feat_tmp = []

        for calculator in self.calculators:
            
            f = calculator(frame)

            if f.keys.names == ['species_center', 'species_neighbor']:
                #supports radial spectrum  

                f = f.keys_to_properties(["species_neighbor"])
                
                feat_tmp.append(f)

            elif f.keys.names == ['species_center', 'species_neighbor_1', 'species_neighbor_2']:
                
                perm = combinations_with_replacement(global_species,2)
                pairs = np.array(list(perm)).reshape(-1,2)
                pairs_comb = equistore.Labels(["species_neighbor_1","species_neighbor_2"], values=pairs)

                f = f.keys_to_properties(["species_neighbor_1","species_neighbor_2"])

                feat_tmp.append(f)
            
            else:
                raise NotImplementedError("The dataset currently only supports radial and radial spectrum features")

        return join(feat_tmp, axis="properties")   

    
    def __init__(
        self,
        frames: Union[ase.Atoms,List[ase.Atoms]],
        calculators: Union[rascaline.torch.calculators.CalculatorModule,\
                           List[rascaline.torch.calculators.CalculatorModule]],
        hypers: Union[dict,List[dict]] = None,
        do_gradients: bool = False,
        do_cell_gradients: bool = False,
        precompute: bool = False,
        on_disk: bool = False,
        lazy_fill_up: bool = True,
        transforms: List[torch.nn.Module] = [],
        memory_save: bool = False,
        energy_key: str = None,
        forces_key: str = None,
        stress_key: str = None,
    ):

        #assert that frames and calculators are not empty
        #TODO:
        # test that assert is raised when frames is empty
        # test that assert is raised when calculators is empty


        """
        if do_gradients:
            assert forces_key is not None
        
        if do_cell_gradients:
            assert stress_key is not None
        
        # ??
        if do_cell_gradients:
            assert do_gradients is True
        
        """

        assert len(frames) > 0

        if isinstance(frames, ase.Atoms):
            frames = [frames]
        
        # TODO: more logic to check for calculators?
        if isinstance(calculators, rascaline.torch.calculators.CalculatorModule):
            #print(type(calculators))
            print("single calculator passed")
            #TODO: write test for this
            calculators = [calculators]
        
        #print(calculators)
        assert len(calculators) > 0
        
        self.do_gradients = do_gradients
        self.do_cell_gradients = do_cell_gradients
        self.precompute = precompute




        self.frames = []
        self.all_species = get_global_unique_species(frames)

        """
        if not self.do_gradients:
            #can there be cell gradients but no position gradients?
            forces_key = None
            stress_key = None

        
        #TODO: the forces key can be zero since the ase_to_tensormap 
        does not need the energy key anymore
        


        if not self.do_cell_gradients:
            stress_key = None
        """

        
        # TODO: make this more general
        #self.properties = [ase_to_tensormap(frame, energy_key, forces_key, stress_key) for frame in frames]
        self.properties = [ase_to_tensormap(frame, energy_key, forces_key, stress_key)
                                         for frame in frames]
        
        
        for frame in frames:
            system = rascaline.torch.systems_to_torch(copy.deepcopy(frame),
                                                     positions_requires_grad=self.do_gradients,
                                                     cell_requires_grad=self.do_cell_gradients,)

            self.frames.append(system)

        #TODO: have a loop that will set compute_gradients to true or false depending on do_gradients
        #PROBLEM: no update_hyperparameters option in rascaline yet.

        #for calculator in calculators:
        #    calculator.update_hyperparameters({"compute_gradients": self.do_gradients})

        self.calculators = calculators
        self.lazy_fill_up = lazy_fill_up
        self.on_disk = on_disk
        self.force_key = forces_key
        self.stress_key = stress_key


        self.calculators = []

        for calculator in calculators:

            self.calculators.append(calculator)

        #TODO: check this
        self.feats = {n : None for n in range(len(self.frames))}

        # check if any of the feat_transformers is a global transformer and needs all feat
        # if so, then precompute all feats
        

        """
        #TODO: implement transformers
        if transforms is not None:
            if any([transformer.is_global for transformer in transforms]):   
                precompute = True
        """




        if memory_save:
            self._predict_full_memory()

        if on_disk:
            #make a tmp directory if it exists already raise error
            #TODO: option to pass default pass
            os.makedirs("./tmp", exist_ok=False)


        # if we could make a feature_handle class that implements .retrieve() and .save()
        # we could handle the precompute and on_disk options in a more elegant way
        # -> could be a database, file, in memory etc.

        if self.precompute:
            for i, frame in enumerate(self.frames):

                feat = self._compute_feats(frame, self.all_species)
                
                if on_disk:
                    #save the feats to disk
                    equistore.save(os.path.join("./tmp", f"feat_{i}.npz"), feat)
                else: 
                    self.feats[i] = feat


        # handle that property is mega large: (do on disk, aswell?)


        

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx: int):
        #this needs to be tested:
        
        #TODO: 
        # dataset with one frame
        # dataset with multiple frames
        # dataset with one frame and precompute but no lazy fill up
        # dataset with multiple frames and precompute but no lazy fill up
        # dataset with one frame and precompute and lazy fill up
        # dataset with multiple frames and precompute and lazy fill up
        # dataset with one frame and no precompute and lazy fill up
        # dataset with multiple frames and no precompute and lazy fill up
        # dataset with one frame and no precompute and no lazy fill up
        # dataset with multiple frames and no precompute and no lazy fill up

        if self.precompute:
            
            if self.on_disk:
                feats = equistore.load(os.path.join("./tmp", f"feat_{idx}.npz"))
            else:
                feats = self.feats[idx]


        else:

            if self.feats[idx] is None:
                feats = self._compute_feats(self.frames[idx],self.all_species)
                
                if self.lazy_fill_up:
                    self.feats[idx] = feats
            
            else:
                feats = self.feats[idx]
        
        #test:
        # one test, with do_gradients: return forces?
        # one test, without do_gradients: return only feats?

        #TODO: remove the densification afterwards

        return feats, self.properties[idx], self.frames[idx]

    # TODO: add a save method
    # PROBLEM: calculators can not yet be pickled

    # TODO: add a load method
    # TODO: add a "report" method that prints out the dataset properties
    
    """
    def __repr__(self):
        super().__repr__()
        # ie add to the repr the number of frames and calculators and a report of the contents
        return f"RascalineAtomisticDataset with {len(self)} frames and {len(self.calculators)} calculators"
    """

    # What properties to extract:
    # only define whether it is a global property or not
    # TODO: write a extract global property function
    # 

    def _predict_full_memory(self):
        #predicts the average storage of all properties in memory
        
        #get the RAM memory of the local machine
        # check if psutil is installed

        try:
            import psutil
        except ImportError:
            raise ImportError("psutil is not installed. Please install psutil to use the _predict_full_memory method")

        #get the RAM memory of the local machine
        mem = psutil.virtual_memory()
        #get the number of frames
        n_frames = len(self)

        # get the feats of the 
        memory_footprint = n_frames * 1.

        if 0.8 * memory_footprint > mem.available:
            raise MemoryError("The predicted memory footprint of the dataset is larger than the available memory on the local machine.\
                              Please set precompute to False")



def _equistore_collate(tensor_maps: List[Tuple[equistore.TensorMap,equistore.TensorMap, rascaline.torch.System]]):
    #TODO: add renumbering of the tensor maps, (idx)

    feats = [tensor_map[0] for tensor_map in tensor_maps]
    properties = [tensor_map[1] for tensor_map in tensor_maps]
    #for tensor_map in tensor_maps: tensor_map[2].positions.detach()
    systems = [tensor_map[2] for tensor_map in tensor_maps]

    #print(type(feats[0]))
    #print(type(feats[0].block(0).values))  
    #print(type(feats[0].block(0).samples))  

    #print(type(properties[0]))

    ##for now do densification on the fly 
     
    return join(feats, axis="samples"), join(properties, axis="samples"), systems
#equistore.join(properties, axis="samples"), systems


def create_rascaline_dataloader(
    frames: Union[ase.Atoms,List[ase.Atoms]],
    calculators: Union[rascaline.torch.calculators.CalculatorModule,List[rascaline.torch.calculators.CalculatorModule]],
    do_gradients: bool = False,
    precompute: bool = False,
    lazy_fill_up: bool = True,
    transforms: List[torch.nn.Module] = None,
    memory_save: bool = False,
    shuffle: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: int = 0,
    worker_init_fn = None,
    multiprocessing_context = None,
    generator = None,
    prefetch_factor = None,
    persistent_workers: bool = False,
    collate_fn = _equistore_collate,
    sampler = None,
    batch_sampler = None,
    dataset_kwargs = None,
    energy_key: str = None,
    forces_key: str = None,
    stress_key: str = None,
    **kwargs,
    ):
    """creates a rascaline dataloader
    """
    dataset = RascalineAtomisticDataset(
        frames,
        calculators,
        do_gradients,
        precompute,
        lazy_fill_up,
        transforms,
        memory_save,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_equistore_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        sampler=sampler,
        batch_sampler=batch_sampler,
        **kwargs,
    )

    return dataloader






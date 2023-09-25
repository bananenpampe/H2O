#from union_join import union_join
#from join import join

from metatensor.torch import join
from metatensor.torch.operations import sort

def union_collate_fn(tensor_maps):
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
     
    return join(feats, axis="samples", different_keys="union"), join(properties, axis="samples"), systems

def metatensor_collate_sort(tensor_maps):
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

    prop_joined = join(properties, axis="samples")
    out_sort = sort(prop_joined, axes="samples")
    feats_joined = join(feats, axis="samples", different_keys="union")
    feats_sorted = sort(feats_joined, axes="samples")

    return feats_sorted, out_sort, systems
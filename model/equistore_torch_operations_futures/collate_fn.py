from union_join import union_join
from join import join


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
     
    return union_join(feats, axis="samples"), join(properties, axis="samples"), systems
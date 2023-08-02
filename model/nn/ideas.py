"""
class EquistoreLinear(torch.nn.Module):
    
    def __init__(self, 
                 n_in: Union[int,Tuple[int,...]], 
                 n_out: Union[int,Tuple[int,...]], 
                 n_blocks: Union[int,Tuple[int,...]]=None,
                 block_keys_in: List[Union[int, Tuple[int,...]]] = None,
                 block_keys_out: List[Union[int, Tuple[int,...]]] = None,
                 block_keys_in_labels: Union[List[str],equistore.Labels] = None,
                 block_keys_out_labels: Union[List[str],equistore.Labels] = None):
        
        super().__init__()    

        if isinstance(n_in, tuple):
            self.n_blocks = len(n_in)    
   
        else:
            self.n_blocks = 1


        if isinstance(n_out, tuple):
            assert len(n_out) == self.n_blocks, "number of blocks in n_out does not match the shapes from n_in"
        else:
            self.n_out = [n_out for i in range(self.n_blocks)]

        assert n_blocks is None or n_blocks == self.n_blocks, "number of blocks does not match the shapes from n_in" 
        assert block_keys_in is None or len(block_keys_in) == self.n_blocks, "number of blocks does not match the number of block keys in" 
        assert block_keys_out is None or len(block_keys_out) == self.n_blocks, "number of blocks does not match the number of block keys out"
        
        if block_keys_in is None:
            self.block_keys_in = [i for i in range(self.n_blocks)]
        else:
            self.block_keys_in = block_keys_in
        
        self.n_out = n_out


        assert len(self.block_keys) == len(self.n_in) == len(self.n_out), "something went wrong with the number of blocks and their keys"

        self.linear = torch.nn.ModuleDict(
            {k: torch.nn.Linear(n_in, n_out) for k, feat_in, feat_out in zip(self.block_keys_in, self.n_in, self.n_out)}
        )
"""
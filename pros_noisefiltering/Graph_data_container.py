import numpy as np


class Graph_data_container:
    def __init__(self, x:np.ndarray, y:np.ndarray, label:str) -> None:
        self.x = x
        self.y = y
        self.label = label
    
    @property 
    def xs_lim(self):
        x_l =  np.floor(np.log10 (max(1, min(self.x)) ))
        x_u =  np.ceil(np.log10 (max(1, max(self.x)) ))
        return [10**x_l, 10**x_u]
    @property 
    def ys_lim(self):
        x_l =  np.floor(np.log10 ( min(self.y) ))-1
        x_u =  np.ceil(np.log10 ( max(self.y) ))+1
        return [10**x_l, 10**x_u]
    
    @property 
    def extrema(self):
        """returns the extreme values for x and y 
        [x_min, x_max, y_min, y_max]
        Returns:
            _type_: _description_
        """        
        return [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

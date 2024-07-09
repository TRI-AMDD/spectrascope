# Description: Utility functions for spectrascope


from multiprocessing import Queue
import numpy as np 

def oxidation_state_worker(queue, struct): 
    """
    Helper function for multiprocessing to calculate oxidation state of a structure and return the output
    Args:
        queue (multiprocessing.Queue): queue to put the return value in
        struct (pymatgen.core.structure.Structure): pymatgen Structure object
    """
    ret = struct.add_oxidation_state_by_guess()
    queue.put(ret)
    
def linear(x, m, b):
    """
    Calculates the linear function y = m*x + b.
    Args:
        x (float): The input value.
        m (float): The slope of the line.
        b (float): The y-intercept of the line.
    Returns:
        float: The calculated value of y.
    """
    return m*x + b


def find_nearest(x, value):
    """
    Find the index of the nearest value in an array.
    Args:
        x (np.array): The array to search.
        value (float): The value to search for.
    Returns:
        int: The index of the nearest value.
    """
    return np.argmin(np.abs(x - value))

def get_mean_nn(struct, nnalgo, el):
    """"
    Calculate the mean nearest neighbor distance for a specific element in a structure
    Args: 
        struct (pymatgen.core.structure.Structure): pymatgen structure object
        nnalgo (pymatgen.analysis.local_env): algorithm to calculate nearest neighbor distances
        el (str): element to calculate nearest neighbor distances for
    Returns: 
        float: mean nearest neighbor distance for the specified element in the structure 
    """
    nn_distances = []
    for i, label in enumerate(struct.labels): 
        if label == el: 
            nn_info = nnalgo.get_nn_info(struct, i)
            for j, neighbor in enumerate(nn_info): 
                nn_distances.append(struct.get_distance(i, neighbor['site_index']))
    return np.nanmean(nn_distances)

def func_unfix_width_reg(eng, *params):
        #TODO: interpolation
        # print(params)
        y = np.zeros_like(eng)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i+1]
            wid = params[i+2]
            y = y + amp * np.exp(-((eng - ctr)**2/(2*(wid/2.355)**2)))
        return y 

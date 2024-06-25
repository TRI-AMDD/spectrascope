# Description: Utility functions for spectrascope


from multiprocessing import Queue

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
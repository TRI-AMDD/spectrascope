"""
Miscellaneous utility functions for SpectraScope
"""

from multiprocessing import Queue
import numpy as np
import pandas as pd

__docformat__ = "numpy"


def load_data(fn):
    """Load data from a file.

    Parameters
    -----------
    fn : str
        The file path.
        
    Returns
    -------
    pd.DataFrame or np.array or dict
        The loaded data.
    """
    
    if fn.endswith("json"):
        df = pd.read_json(fn)
        with open(fn) as f:
            data = json.load(f)
    elif fn.endswith("csv"):
        data = pd.read_csv(fn)
    elif fn.endswith("npy"):
        data = np.load(fn)
    return data


def oxidation_state_worker(queue, struct):
    """Helper function for add_oxidation_state_by_guess taking too long. 
    
    Calculate the oxidation state of a structure and return the output;
    to deal with possible long calculations.

    Parameters
    -----------
        queue : multiprocessing.Queue
            queue to put the return value in.
        struct: pymatgen.core.structure.Structure
            pymatgen Structure object.
    """
    
    ret = struct.add_oxidation_state_by_guess()
    queue.put(ret)


def linear(x, m, b):
    """
    Calculates the linear function y = m*x + b.

    Parameters
    ----------
    x : float
        The input value.
    m : float
        The slope of the line.
    b : float
        The y-intercept of the line.

    Returns
    -------
    float
        The calculated value of y.
    """
    
    return m * x + b


def find_nearest(x, value):
    """Find the index of the nearest value in an array.
    
    Parameters
    ----------
    x : np.array
        The array to search.
    value : float
        The value to search for.
    
    Returns
    -------
    int
        The index of the nearest value.
    """
    
    return np.argmin(np.abs(x - value))


def get_mean_nn(struct, nnalgo, el):
    """Calculate the mean nearest neighbor distance for a specific element in a structure
    
    Parameters
    ----------
    struct : pymatgen.core.structure.Structure
        pymatgen structure object.
    nnalgo : pymatgen.analysis.local_env
        algorithm to calculate nearest neighbor distances.
    el : str
        element to calculate nearest neighbor distances for.
    
    Returns
    -------
    float
        mean nearest neighbor distance for the specified element in the structure.
    """
    
    nn_distances = []
    for i, label in enumerate(struct.labels):
        if label == el:
            nn_info = nnalgo.get_nn_info(struct, i)
            for j, neighbor in enumerate(nn_info):
                nn_distances.append(struct.get_distance(i, neighbor["site_index"]))
    return np.nanmean(nn_distances)


def func_unfix_width_reg(eng, *params):
    """Helper function for gaussian peak fitting and plotting.
    
    Parameters
    ----------
    eng : np.array
        energy values.
    *params : list
        parameters from the gaussian peak fitting.

    Returns
    y : np.array 
        fit values across the energy range.
    """
    
    y = np.zeros_like(eng)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y = y + amp * np.exp(-((eng - ctr) ** 2 / (2 * (wid / 2.355) ** 2)))
    return y


def NormalizeData(arr):
    """Normalize array between 0, 1.

    Parameters
    ----------
    arr : np.array or list 
        input array.
        
    Returns
    -------
    np.array
        input array normalized between 0, 1.
    """
    
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def sort_get_value(my_dict: dict, id2: int = 20, id1: int = 0):
    """Sort a dictionary by values and return the values between id1 and id2.

    Parameters
    ----------
    my_dict : dict
        dictionary to sort.
    id2 : int, optional
        Last index. Defaults to 20.
    id1 : int, optional

    Returns
    -------
    labels : list
        sorted labels in the range id1 to id2.
    values : list
        sorted values in the range id1 to id2.
    """
    
    labels = np.array([str(l) for l in list(my_dict.keys())])
    values = np.array(list(my_dict.values()))
    sorted_values = np.argsort(values)[::-1]

    return labels[sorted_values][id1:id2], values[sorted_values][id1:id2]


def tamimoto(s1: np.array, s2: np.array):
    """Tamimoto similarity index between two binary (T/F or 0/1) arrays s1 and s2.
    
    The tamimoto coefficient is a ratio that measures the similarity between two sets of data.
    
    Parameters
    ----------
    s1 : np.array
        binary array.
    s2 : np.array
        binary array.
        
    Returns
    -------
    J : np.array
        Tamimoto similarity index.
    """
    
    # If s2 is a single array, make it a 2D array
    if len(s2.shape) == 1:
        s2 = np.expand_dims(s2, axis=0)

    m11 = ((s1 == 1) & (s2 == 1)).sum(axis=1)
    m00 = ((s1 == 0) & (s2 == 0)).sum(axis=1)
    m01 = ((s1 == 0) & (s2 == 1)).sum(axis=1)
    m10 = ((s1 == 1) & (s2 == 0)).sum(axis=1)
    J = m11 / (m01 + m10 + m11)
    dist = 1 - J
    return J


def entropy(p0, p1):
    """Calculate entropy of a random variable. 
    
    Entropy of a random variable is the avarage level of "information", "surprise", or
    "uncertainty" inherent in the variable's possible outcomes.

    Parameters
    ----------
    p0 : float
        The probability of choosing class 1.
    p1 : float
        The probability of choosing class 2.
        
    Returns
    -------
    float
        entropy of random variable.
    """
    
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def arr_entropy(arr):
    """Calculate the entropy of a binary array.

    Parameters
    ----------
    arr : np.array or pd.DataFrame
        binary array or pd.DataFrame.

    Returns
    -------
    entropy : float
        entropy of the array.
    """
    
    Ny = arr.shape[0]
    p0 = arr.sum(axis=0) / Ny
    p1 = (Ny - arr.sum(axis=0)) / Ny
    return entropy(p0, p1)


def LCEN_selected_features(model, top_features: int):
    """Return the selected features for a fitted LCEN model.

    Parameters
    ----------
    model : dict
        Fitted LCEN model from main.SPA.
    top_features : int
        The number of top X features to return.
        
    Returns
    -------
    vals : list
        The value of the top "top_features" features.
    """
    
    params = model["model_params"]
    num_selected = len(params)
    par_names = np.array(params)[:, 0]
    par_locs = np.array(
        [
            p.split("^")[0].replace("x", "") if p != "intercept" else "intercept"
            for p in par_names
        ]
    )
    par_vals = np.array(params)[:, 1].astype("float64")
    par_vals_abs = np.abs(par_vals)
    importance = np.argsort(par_vals_abs)[::-1]

    selected_features = []
    vals = []
    for loc, val in zip(
        par_locs[importance][:top_features], par_vals[importance][:top_features]
    ):
        if loc == "intercept":
            pass
        else:
            selected_features.append(int(loc))
            vals.append(val)

    return vals, selected_features

def maximal_correlation_from_ace(X, y):
    """Calculate the maximal correlation between X and y using ACE.
    
    Parameters
    ----------
    X : np.array
        The input array.
    y : np.array
        The target variable.
    """
    if 'model' not in globals():
        from ace import model
    if 'pearsonr' not in globals():
        from scipy.stats import pearsonr
        
    # 1) Convert to the format this ACE wants: list of 1-D arrays
    x_list = [X[:, j] for j in range(X.shape[1])]

    # 2) Build ACE model from X and y
    myace = model.Model()
    myace.build_model_from_xy(x_list, y)  # runs ACE internally

    # 3) Grab learned transforms
    x_transforms = myace.ace.x_transforms   # list of length M, arrays of shape (N,)
    y_transform = myace.ace.y_transform     # array of shape (N,)

    # 4) Sum predictor transforms across features for each sample
    # theta(Y) is calculated based on the sum of the x_transforms 
    sum_phi = np.sum(np.column_stack(x_transforms), axis=1)  # shape (N,)

    # 5) Pearson correlation between sum(phi_j(X_j)) and theta(Y)
    r, _ = pearsonr(sum_phi, y_transform)

    # HGR maximal correlation is nonnegative by definition; take abs
    return abs(r)
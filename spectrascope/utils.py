# Description: Utility functions for spectrascope


from multiprocessing import Queue
import numpy as np

x_domains_xanes_torrisi = {
    "Ti": np.linspace(4970, 5021.0, 100),
    "V": np.linspace(5468.0, 5520.6, 100),
    "Cr": np.linspace(5993.1, 6045.7, 100),
    "Mn": np.linspace(6541.7, 6594.4, 100),
    "Fe": np.linspace(7115.0, 7167.8, 100),
    "Co": np.linspace(7713.5, 7765.8, 100),
    "Ni": np.linspace(8336.5, 8388.7, 100),
    "Cu": np.linspace(8987.5, 9039.7, 100),
}

x_domains_xanes_billinge = {
    "Ti": np.linspace(4969.0, 5021.0, 200),
    "V": np.linspace(5468.671, 5520.63, 200),
    "Cr": np.linspace(5993.195, 6045.686, 200),
    "Mn": np.linspace(6542.896, 6594.431, 200),
    "Fe": np.linspace(7115.617, 7167.759, 200),
    "Co": np.linspace(7714.284, 7765.838, 200),
    "Ni": np.linspace(8337.207, 8388.723, 200),
    "Cu": np.linspace(8988.142, 9039.728, 200),
}

x_domain_pdf_billinge = np.linspace(0, 10, 200)


def load_data(fn):
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
    return m * x + b


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
    """ "
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
                nn_distances.append(struct.get_distance(i, neighbor["site_index"]))
    return np.nanmean(nn_distances)


def func_unfix_width_reg(eng, *params):
    # TODO: interpolation
    # print(params)
    y = np.zeros_like(eng)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y = y + amp * np.exp(-((eng - ctr) ** 2 / (2 * (wid / 2.355) ** 2)))
    return y


def NormalizeData(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def sort_get_value(my_dict, id2=20, id1=0):
    labels = np.array([str(l) for l in list(my_dict.keys())])
    values = np.array(list(my_dict.values()))
    sorted_values = np.argsort(values)[::-1]

    return labels[sorted_values][id1:id2], values[sorted_values][id1:id2]


def tamimoto(s1, s2):
    m11 = ((s1 == 1) & (s2 == 1)).sum(axis=1)
    m00 = ((s1 == 0) & (s2 == 0)).sum(axis=1)
    m01 = ((s1 == 0) & (s2 == 1)).sum(axis=1)
    m10 = ((s1 == 1) & (s2 == 0)).sum(axis=1)
    J = m11 / (m01 + m10 + m11)
    dist = 1 - J
    return J


def entropy(p0, p1):
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def arr_entropy(arr):
    Ny = arr.shape[0]
    p0 = arr.sum(axis=0) / Ny
    p1 = (Ny - arr.sum(axis=0)) / Ny
    return entropy(p0, p1)

def LCEN_selected_features(model, top_features): 
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
    for loc, val in zip(par_locs[importance][:top_features], par_vals[importance][:top_features]):
        if loc == "intercept":
            pass
        else:
            selected_features.append(int(loc))
            
    return selected_features
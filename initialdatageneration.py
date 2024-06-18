import numpy as np 
import pandas as pd
from mp_api.client import MPRester
from emmet.core.xas import Edge, XASDoc, Type
from pymatgen.analysis.local_env import CrystalNN
from spectrascope.features import spectral_features


# TODO: Figure out how to pull xas and structure information with the same query, right now not doing that 
# TODO: Add a progress bar to the function
# TODO: figure out how to cancel calculation if it is taking too long
# TODO: MP has multiple spectrum per structure, and is returning a linear combination here, need to think about

mykey = 'HrWPbvhi2NRaOjrofHmVEOcCi8ElPurB'

def get_data(mykey, elements=['Ti', 'Ni'], subset=-1, cnalgo=CrystalNN()): 
    """
    Retrieves XAS (X-ray Absorption Spectroscopy) documents for specified
    elements using the Materials Project API.
    It creates a dataframe consisting of the XAS docs information, the structure (pymatgen.core.structure.Structure),
    the oxidation number calculated by pymatgen.core.structure.Structure.add_oxidation_state_by_guess(), and the 
    coordination number calculated by pymatgen.analysis.local_env.CrystalNN algorithm, taking the guessed
    oxidation state into account 
    
    Args: 
        mykey (str): Materials Project API key (https://materialsproject.org/open)
        elements (list): elements to get spectra of 
        subset (int): number of documents to retrieve (default=-1, all documents)
        cnalgo (pymatgen.analysis.local_env): algorithm to calculate coordination number (default=CrystalNN(),good option:VoronoiNN(tol=0.5))
    Returns: 
        data (pd.DataFrame) : index: materials project id 
    """
    
    a  = MPRester(mykey)

    with MPRester(mykey,use_document_model=False, monty_decode=False) as mpr: 
        data = {}
        for element in elements: 
            docs = mpr.materials.xas.search(edge=Edge.K, 
                            spectrum_type=Type.XANES, 
                            absorbing_element=element,
                            fields=["material_id", "xas_id", "spectrum", "formula_pretty"])
            for doc in (docs[:subset]): 
                mpid = doc['material_id']
                struct = a.get_structure_by_material_id(mpid)
                try: 
                    if struct == []: 
                        print(f'{mpid} has xas but no structure information')
                        pass
                    else: 
                        cn_arr = []
                        oxidation_arr = []
                        for n, label in enumerate(struct.labels):    
                            if label == element:                 
                                    oxidation = struct.add_oxidation_state_by_guess()[n].specie.oxi_state
                                    cn = cnalgo.get_cn(struct,n)
                                    if cn not in cn_arr: 
                                        cn_arr.append(cn)
                                    if oxidation not in oxidation_arr: 
                                        oxidation_arr.append(oxidation)            
                        doc['structure'] = struct 
                        doc['CN'] = cn_arr
                        doc['CN_mean'] = np.mean(cn_arr)
                        doc['oxidation'] = oxidation_arr
                        xas = doc['spectrum']
                        temp = spectral_features(xas['x'], xas['y'])
                        temp.pointwise_features(ne=100)
                        temp.polynomial_featuresall(interpolate=True, Narr=np.array([4,5,10,20]))
                        doc['spectrum_point'] = temp.yinterp # gives (array (array ))
                        doc['spectrum_poly'] = temp.polyfeatures_all
                        data[mpid] = doc
                except: 
                    pass 
                
    
    data = pd.DataFrame(data).T
    return data 

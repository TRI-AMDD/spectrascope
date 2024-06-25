
import numpy as np 
import pandas as pd
from mp_api.client import MPRester
from emmet.core.xas import Edge, XASDoc, Type
from pymatgen.analysis.local_env import CrystalNN
from spectrascope.features import spectral_features
import json
from tqdm import tqdm 
import time
from multiprocessing import Process, Queue
from spectrascope.utils import oxidation_state_worker 


# TODO: Figure out how to pull xas and structure information with the same query, right now not doing that 
# TODO: MP has multiple spectrum per structure, and is returning a linear combination here, need to think about
# TODO: Add mean nearest neighbor distance calculation 

def get_data(mykey, fn, elements=['Ti', 'Ni'], subset=-1, cnalgo=CrystalNN(), chunks=50, add_oxidation_state=False, has_oxygen=True, 
             nn_algo=CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)): 
    """
    Retrieves XAS (X-ray Absorption Spectroscopy) documents for specified elements using the Materials Project API.
    It creates a dictionary consisting of the XAS docs information, the structure (pymatgen.core.structure.Structure.as_dict()) the
    coordination number calculated by cnalgo, a pymatgen.analysis.local_env.CrystalNN algorithm, taking the guessed
    oxidation state into account 
    
    This dictionary is saved to a json file every 'chunks' number of entries.
    
    Args: 
        mykey (str): Materials Project API key (https://materialsproject.org/open)
        fn (str): filename to save data to; will be written over every 'chunks' number of entries
        elements (list): elements to get spectra of 
        subset (int): number of documents to retrieve (default=-1, all documents)
        cnalgo (pymatgen.analysis.local_env): algorithm to calculate coordination number (default=CrystalNN(),good option:VoronoiNN(tol=0.5))
        chunks (int): number of entries to save to file at a time (default=50)
        add_oxidation_state (bool): whether to add oxidation state to the dictionary (default=False); necessary if CrystalNN() cn algo is used
        has_oxygen (bool): whether to only return spectra that have at least one oxygen atom in structure (default=True)
        nn_algo (pymatgen.analysis.local_env): algorithm to calculate mean nearest neighbor distance (default=CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False))
    Returns: 
        data (dict) : dictionary of information 
    """
    
    a  = MPRester(mykey,  mute_progress_bars=True)
    counter = 0        
    data = {}
    
    if cnalgo == CrystalNN():
        assert(add_oxidation_state == True), 'Cannot use CrystalNN algorithm with add_oxidation_state=False, please use a different algorithm or set add_oxidation_state=True'
        
    with MPRester(mykey,use_document_model=False, monty_decode=False, mute_progress_bars=False) as mpr:
        
        for element in elements: 
            docs = mpr.materials.xas.search(edge=Edge.K, 
                            spectrum_type=Type.XANES, 
                            absorbing_element=element,
                            elements = [el for el in [element, 'O'*has_oxygen] if el != ''],
                            fields=["material_id", "xas_id", "spectrum", "formula_pretty"])
            
            for doc in tqdm(docs[:subset]): 
                mpid = doc['material_id']
                struct = a.get_structure_by_material_id(mpid)   
                # nn_info = nn_algo.get_all_nn_info(struct) # get the nn info for each el site in structure 
                
                try: # not sure if this try is still necessary actually  
                    # sometimes query returns empty structure
                    if struct == [] or np.min(doc['spectrum']['y']) < 0: # if structure is empty or spectrum is negative
                        pass
                    else: 
                        cn_arr = []
                        nn_arr = []
                        if add_oxidation_state:
                            oxidation_arr = []
                            # Sometimes guessing the oxidation state in this way gets stuck indefinitely, so create a Process to cancel it if it takes too long
                            q = Queue()
                            p = Process(target=oxidation_state_worker, args=(q, struct))
                            p.start()
                            time.sleep(3) # determined as how long it takes for regular oxidation state calculation to finish
                            if p.is_alive():
                                p.terminate()
                                p.join()
                                continue 
                            else: 
                                struct = q.get()
                                for n, label in enumerate(struct.labels):    
                                    if label == element:                 
                                        cn = cnalgo.get_cn(struct,n)
                                        # nn_arr.append()
                                        # for i, site in enumerate(nn_info): 
                                            # nn_arr.append(struct.get_distance(Ni, info[i]['site_index'])) # get distance of nn for each 
                                        oxidation = struct[n].specie.oxi_state
                                        if cn not in cn_arr: 
                                            cn_arr.append(cn)
                                        if oxidation not in oxidation_arr: 
                                            oxidation_arr.append(oxidation) 
                                        doc['oxidation'] = oxidation_arr
                        else: 
                            for n, label in enumerate(struct.labels):
                                if label == element: 
                                    cn = cnalgo.get_cn(struct,n)
                                    if cn not in cn_arr: 
                                        cn_arr.append(cn) 
                                        
                        doc['structure'] = struct.as_dict() 
                        doc['CN'] = cn_arr
                        xas = doc['spectrum']
                        temp = spectral_features(xas['x'], xas['y'])
                        temp.pointwise_features(ne=100)
                        temp.polynomial_featuresall(interpolate=True, Narr=np.array([4,5,10,20]))
                        doc['spectrum_point'] = temp.yinterp.tolist() # gives (array (array ))
                        doc['spectrum_poly'] = temp.polyfeatures_all.tolist()
                        
                        data[mpid] = doc
                        
                        if counter % chunks == 0 or counter == (subset-1):
                            with open(fn, 'w') as f: 
                                json.dump(data,f) 
                    
                except OSError: 
                    pass
                
                counter += 1
                
    print(f'Data saved to {fn} with {len(data.keys())} entries. structure information saved as a dictionary in data[mpid][\'structure\'].\n',
          'To convert structure information back to pymatgen.core.structure.Structure, use data[\'structure\'] = data[\'structure\'].apply(lambda x: Structure.from_dict(x))')      
    if add_oxidation_state:
        print('oxidation state added by Structure.add_oxidation_state_by_guess()')  
    return data

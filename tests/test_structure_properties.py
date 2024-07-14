from spectrascope.structure_property_featurization import *
import os
from pymatgen.core import Structure

# Load in the files in test_files into a list of pymatgen structures
# Do so by obtaining the list of .json files in test_files
# Then, load in the structures from the .json files


test_files = 'tests/test_files/'  # Replace with the actual path to the test files directory

structure_list = []

for file_name in os.listdir(test_files):
    if file_name.endswith('.json'):
        file_path = os.path.join(test_files, file_name)
        structure = Structure.from_file(file_path)
        structure_list.append(structure)


def test_is_element_present():
    # Test if the function is_element_present works as expected
    # Test if the function returns True if the element is present in the structure
    # Test if the function returns False if the element is not present in the structure

    not_present='Og'

    for structure in structure_list:
        for element in structure.composition.elements:
            assert is_element_present(structure, element) == True
            assert is_element_present(structure, str(element)) == True

        assert is_element_present(structure, not_present) == False
    

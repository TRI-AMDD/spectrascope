from spectrascope.structure_property_featurization import *
import os
from pymatgen.core import Structure

# Load in the files in test_files into a list of pymatgen structures
# Do so by obtaining the list of .json files in test_files
# Then, load in the structures from the .json files


test_files = (
    "test_files/"  # Replace with the actual path to the test files directory
)

structure_list = []
comp_to_structure = {}

for file_name in os.listdir(test_files):
    if file_name.endswith(".json"):
        file_path = os.path.join(test_files, file_name)
        structure = Structure.from_file(file_path)
        structure_list.append(structure)

for structure in structure_list:
    comp_to_structure[structure.composition] = structure


def test_is_element_present():
    # Test if the function is_element_present works as expected
    # Test if the function returns True if the element is present in the structure
    # Test if the function returns False if the element is not present in the structure

    not_present = "Og"

    for structure in structure_list:

        for element in structure.composition.elements:
            assert is_element_present(structure, element) == True

        assert is_element_present(structure, not_present) == False


def test_is_cn_present():

    copper_structure = comp_to_structure[Composition(("Cu") * 4)]

    # Test if the function is_cn_present works as expected

    results = {}
    for i in range(1, 13):
        results[i] = is_cn_present(copper_structure, i)

    for i in range(1, 12):
        assert not results[i]
    assert results[12]


def test_is_element_cn_present():

    copper_structure = comp_to_structure[Composition(("Cu") * 4)]

    # Test if the function is_cn_present works as expected

    assert is_element_cn_present(copper_structure, "Cu", 4) == False
    assert is_element_cn_present(copper_structure, "Cu", 6) == False
    assert is_element_cn_present(copper_structure, "Cu", 12) == True

    assert is_element_cn_present(copper_structure, "O", 6) == False


def test_is_nn_distance_present():

    copper_structure = comp_to_structure[Composition(("Cu") * 4)]

    # Test if the function is_nn_distance_present works as expected

    # nn distances are all non-diagonal elements of the distance matrix
    nn_distances = copper_structure.distance_matrix[
        np.triu_indices(len(copper_structure), k=1)
    ]

    tolerance = 0.01
    for dist in nn_distances:
        assert (
            is_nn_distance_present(copper_structure, dist - tolerance, dist + tolerance)
            == True
        )

    # Use an unphysically small NN distance to test non-existence of an nn distance
    assert not is_nn_distance_present(copper_structure, 0.01, 0.02)


def test_identify_crystal_system():

    copper_structure = comp_to_structure[Composition(("Cu") * 4)]

    sys = identify_crystal_system(structure=copper_structure)

    assert identify_crystal_system(structure=copper_structure)["is_cubic"]
    wrong_structures = [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
    ]

    for wrong_structure in wrong_structures:
        assert not identify_crystal_system(structure=copper_structure)[
            f"is_{wrong_structure}"
        ]


def test_is_element_nn_distance_present():

    # todo ensure robustness against species; oxidation states
    # todo write
    pass


def test_is_in_space_group():

    copper_structure = comp_to_structure[Composition(("Cu") * 4)]

    assert is_in_space_group(copper_structure, 225)
    assert not is_in_space_group(copper_structure, 224)


def is_crystal_system():

    copper_structure = comp_to_structure[Composition(("Cu") * 4)]

    assert is_crystal_system(copper_structure, "cubic")
    assert not is_crystal_system(copper_structure, "hexagonal")

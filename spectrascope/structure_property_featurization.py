from pymatgen.core import Structure, Composition, Element, Species
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
import numpy as np
import pandas as pd
from typing import Union


def is_element_present(
    structure,
    element: Union[str, Element],
    ignore_oxidation_state=True,
    return_label=False,
):
    composition = structure.composition

    if isinstance(element, str):
        element = Element(element)
    if isinstance(element, Species):
        element = element.element
    if ignore_oxidation_state:
        elements = composition.chemical_system.split("-")
        elements = [Element(e) for e in elements]
        value = element in elements
    else:
        raise NotImplementedError

    if return_label:
        label = "has_" + element.symbol
        return value, label
    else:
        return value


def is_cn_present(structure: Structure, cn, indices=None, return_label=False):
    cnalgo = CrystalNN()
    cn_arr = []
    if indices is None:
        indices = range(len(structure))
    for n in indices:
        cur_cn = cnalgo.get_cn(structure, n)
        if cur_cn not in cn_arr:
            cn_arr.append(cur_cn)
    value = cn in cn_arr
    if return_label:
        label = "has_CN_" + str(cn)
        return value, label
    else:
        return value


def is_element_cn_present(
    structure: Structure, element: Union[str, Element], cn, return_label=False
):
    element_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == element
    ]

    if len(element_indices) == 0:
        return False

    value, label = is_cn_present(
        structure, cn, indices=element_indices, return_label=True
    )
    if return_label:
        label = element + "_has_CN_" + str(cn)
        return value, label
    else:
        return value


def is_nn_distance_present(
    structure: Structure,
    nn_distance_lower,
    nn_distance_upper,
    indices=None,
    return_label=False,
):

    nn_algo = CrystalNN()
    if indices is None:
        indices = range(len(structure))
    value = False
    for n in indices:
        nn_info = nn_algo.get_nn_info(structure, n)
        for j, neighbor in enumerate(nn_info):
            distance = structure.get_distance(n, neighbor["site_index"])
            if nn_distance_lower <= distance <= nn_distance_upper:
                value = True
                break
        if value:
            break
    if return_label:
        label = "nn_in_[" + str(nn_distance_lower) + "," + str(nn_distance_upper) + "]"
        return value, label
    else:
        return value


def is_element_nn_distance_present(
    structure: Structure,
    element: Union[str, Element],
    nn_distance_lower,
    nn_distance_upper,
    return_label=False,
):
    element_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == element
    ]
    value, label = is_nn_distance_present(
        structure,
        nn_distance_lower,
        nn_distance_upper,
        indices=element_indices,
        return_label=True,
    )
    if return_label:
        label = (
            element
            + "_nn_in_["
            + str(nn_distance_lower)
            + ","
            + str(nn_distance_upper)
            + "]"
        )
        return value, label
    else:
        return value


def is_oxidation_state_present(
    structure: Structure, oxidation_state, indices=None, return_label=False
):
    oxidation_arr = []
    if indices is None:
        indices = range(len(structure))
    for n in indices:
        oxidation = structure[n].specie.oxi_state
        if oxidation not in oxidation_arr:
            oxidation_arr.append(oxidation)
    value = oxidation_state in oxidation_arr
    if return_label:
        label = "has_oxidation_" + str(oxidation_state)
        return value, label
    else:
        return value


def is_element_oxidation_state_present(
    structure: Structure,
    element: Union[str, Element],
    oxidation_state,
    return_label=False,
):
    element_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == element
    ]
    value, label = is_oxidation_state_present(
        structure, oxidation_state, indices=element_indices, return_label=True
    )
    if return_label:
        label = element + "_has_oxidation_" + str(oxidation_state)
        return value, label
    else:
        return value


def identify_crystal_system(structure: Structure):
    crystal_structures = [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    ]
    crystal_system = SpacegroupAnalyzer(structure).get_crystal_system()
    results = {}
    for crystal_structure in crystal_structures:
        value = crystal_system == crystal_structure
        label = "is_" + crystal_structure
        results[label] = value
    return results


def is_crystal_system(structure: Structure, crystal_system: str, return_label=False):
    crystal_systems = identify_crystal_system(structure)

    value = crystal_systems.get("is_" + crystal_system, False)

    if return_label:
        label = "is_" + crystal_system
        return value, label
    else:
        return value


def is_in_space_group(
    structure: Structure, space_group: Union[int, str], return_label=False
):
    analyzer = SpacegroupAnalyzer(structure)
    if isinstance(space_group, int):
        value = analyzer.get_space_group_number() == space_group
    elif isinstance(space_group, str):
        value = analyzer.get_space_group_symbol() == space_group
    else:
        raise TypeError("space_group must be an integer or a string")

    if return_label:
        label = "is_" + str(space_group)
        return value, label
    else:
        return value


def has_symmetry(structure: Structure, symmetry_operation, return_label=False):
    raise NotImplementedError

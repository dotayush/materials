"""
"materials." is a collection of post-processing scripts to process and
anaylze materials data from Quantum ESPRESSO (QE) calculations.
Copyright (C) 2025 ayush.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os


import matplotlib
import numpy as np
import pandas as pd
from itertools import product
from typing import Union
from ase.atoms import Atoms
from ase.calculators.espresso import EspressoProfile
from ase.io import write
from mp_api.client import MPRester
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from shared import (
    FUNCTIONAL,
    MP_API_KEY,
    SAVED_STRUCTURES_DIR,
    PSEUDOPOTENTIALS_DIR,
    PROJECT_NAME,
    setup_output_project
)

# matplotlib backend for plotting
matplotlib.use("Qt5Agg")
espresso_profile = EspressoProfile(
    command="mpirun -np 4 pw.x", pseudo_dir=PSEUDOPOTENTIALS_DIR
)

QEInputType = dict[str, Union[str, float, int]]


def print_structure(structures, title=None):
    """Print the structure data in a tabular format."""
    structure_data = []
    for structure in structures:
        sga = SpacegroupAnalyzer(structure)
        conventional = sga.get_conventional_standard_structure()
        # lattice_pos: a, b, c, alpha, beta, gamma
        # reciprocal_lattice_pos: a_r, b_r, c_r, alpha_r, beta_r, gamma_r
        if isinstance(structure, Structure):
            structure_data.append(
                {
                    "formula": structure.composition.formula,
                    "space_group": sga.get_space_group_symbol(),
                    "space_group_number": sga.get_space_group_number(),
                    "a": conventional.lattice.a,
                    "b": conventional.lattice.b,
                    "c": conventional.lattice.c,
                    "alpha": conventional.lattice.alpha,
                    "beta": conventional.lattice.beta,
                    "gamma": conventional.lattice.gamma,
                    "a_r": conventional.lattice.reciprocal_lattice.a,
                    "b_r": conventional.lattice.reciprocal_lattice.b,
                    "c_r": conventional.lattice.reciprocal_lattice.c,
                    "alpha_r": conventional.lattice.reciprocal_lattice.alpha,
                    "beta_r": conventional.lattice.reciprocal_lattice.beta,
                    "gamma_r": conventional.lattice.reciprocal_lattice.gamma,
                    "density": [structure.density],
                },
            )

    if title:
        print(f"\n{title}\n{'-' * 30}")
    print(pd.DataFrame(structure_data))
    print("-" * 30 + "\n")


def get_structure_from_id(mp_id: str) -> Structure:
    """Retrieve a structure from the Materials Project database using its ID."""
    # check under ./saved_structures if the structure is already saved
    structure_file = f"{mp_id}.cif"
    structure_path = os.path.join(SAVED_STRUCTURES_DIR, structure_file)
    print(
        f"info: looking for structure {mp_id} under {os.path.relpath(SAVED_STRUCTURES_DIR, os.getcwd())}..."
    )
    if os.path.exists(structure_path):
        print(f"info: structure {structure_file} found in saved structures.")
        return Structure.from_file(structure_path)
    else:
        print(
            f"info: structure {mp_id} not found in saved structures. fetching from Materials Project..."
        )
        if not MP_API_KEY:
            raise ValueError(
                "error: MP_API_KEY is not set and get_structure_from_id was called. please set the api key in the .env file to utilise mp rester client for material retrieval."
            )
        else:
            mpr = MPRester(MP_API_KEY)
        structure = mpr.get_structure_by_material_id(mp_id)

        # ensure saved structures directory exists
        os.makedirs(SAVED_STRUCTURES_DIR, exist_ok=True)

        if isinstance(structure, Structure):
            print(
                f"info: saving structure {structure_file} to {os.path.relpath(SAVED_STRUCTURES_DIR, os.getcwd())}"
            )
            _ = structure.to_file(filename=structure_path, fmt="cif")
            return structure
        else:
            # well, this should never happen, but just in case
            raise ValueError(
                f"error: expected a single Structure object from mp rester, but got {type(structure)}"
            )


# todo: unused but implemented
def substrate_compatibility(structure_a: Structure, structure_b: Structure):
    """Calculate the match of two structures as a substrate and film pair."""
    analyzer = SubstrateAnalyzer()
    matches = analyzer.calculate(structure_a, structure_b)
    if not matches:
        print("info: no suitable substrate matches found")
        return None

    # return the match with the lowest von mises strain
    return min(matches, key=lambda x: x.strain.von_mises_strain)

# todo: unused but implemented
def find_optimal_scale(required_scale: float, max_scale: int = 10):
    """Find scale factors (i, j, k) such that:
    - i * j * k >= required_scale
    - the product is as close as possible to required_scale
    - i, j, k are as balanced (close to each other) as possible
    """
    best_combo = (1, 1, 1)
    best_score = float("inf")

    for i, j, k in product(range(1, max_scale + 1), repeat=3):
        product_val = i * j * k
        if product_val >= required_scale:
            # spread penalty: how unbalanced the values are
            spread = max(i, j, k) - min(i, j, k)
            # closeness to required scale
            excess = product_val - required_scale
            # score combines both (weight spread more heavily to prioritize balance)
            score = spread * 10 + excess
            if score < best_score:
                best_score = score
                best_combo = (i, j, k)

    return best_combo

def create_doped_structure(
    structure: Structure,
    host_element: str,
    dopant_element: str,
    dopant_concentration: float,
):
    conventional = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    c_nat = len(
        [site for site in conventional if site.species_string == host_element]
    )  # 8 Si atoms
    t_nat = c_nat  # let t_nat start from the conventional atoms = 8
    while True:
        if (
            t_nat * dopant_concentration < 1
        ):  # if 8 * (1.1 / 100) <= 1 => 0.088 < 1, then t_nat++
            t_nat += 1
        else:  # if 91 * (0.1 / 100) >= 1 => 1.001 >= 1, then we can stop; here t_nat = 91
            break

    # 91 / 8 = 11.375 (required scale factor)
    required_scale = t_nat / c_nat
    scale = find_optimal_scale(required_scale)
    # if (dopant_concentration*100) <= 5.0:
    #     scale = find_optimal_scale(required_scale)
    # else:
    #     scale_num = max(1, math.ceil(1/(dopant_concentration) ** (1/3)))
    #     scale = (scale_num, scale_num, scale_num)

    # make supercell
    supercell = conventional * scale
    s_nat = len([site for site in supercell if site.species_string == host_element])

    # random substitution
    host_indices = [i for i, _ in enumerate(supercell)]
    np.random.shuffle(host_indices)
    for idx in host_indices[:1]:
        supercell.replace(idx, dopant_element)

    # wrap the supercell to ensure all sites are within the unit cell
    # supercell = supercell.get_sorted_structure()
    # wrapped_supercell = supercell.copy()
    # wrapped_supercell = wrapped_supercell.make_supercell(
    #     np.diag([1, 1, 1]),
    # )

    data_str = f"{scale}({scale[0] * scale[1] * scale[2]})_{t_nat}/{c_nat}_{required_scale:.3f}_{len(supercell)}_{(1 / s_nat) * 100:.3f}"

    return supercell, data_str


def generate_espresso_input(
        struct: Structure, out_dir: str, prefix: str, like: str, xc: str, kpts=None, nbnd=None
):
    atoms = AseAtomsAdaptor.get_atoms(struct)
    if not isinstance(atoms, Atoms):
        raise ValueError(
            f"error: expected a single Atoms object, but got {type(atoms)}"
        )
    atoms.wrap()
    pseudopotentials = {
        s: f"{s.lower()}.UPF" for s in set(atoms.get_chemical_symbols())
    }
    nat = len(atoms)

    # base parameters
    control: QEInputType = {
        "prefix": prefix,  # use the base name of the cif file as prefivx
        "pseudo_dir": PSEUDOPOTENTIALS_DIR,
        "outdir": out_dir,
        "verbosity": "high", # this  is required to extract occupation numbers, else bandgap calculation fails
        "tstress": True,
        "tprnfor": True,
    }
    system: QEInputType = {
        "nbnd": nbnd or 32,
        "nat": nat,
        "ecutwfc": 30,
        "ecutrho": 120,
    }
    electrons: QEInputType = {
        "conv_thr": 1e-8,
        "mixing_beta": 0.4,
    }


    # update base paramters
    if like == "metal":
        system.update({"smearing": "mv", "degauss": 0.01, "occupations": "smearing"})
    elif like == "insulator":
        system.update({"occupations": "fixed"})
    if xc:
        system.update({"input_dft": xc.upper()})  # set exchange-correlation functional
    else:
        system.update({"input_dft": "PBE"})

    # ---- handle SCF input generation ----
    control.update(
        {
            "calculation": "scf",  # set calculation type to SCF
            "wf_collect": True,  # collect wavefunctions for NSCF
        }
    )
    input_file = os.path.join(out_dir, "scf.in")
    write(
        input_file,
        atoms,
        format="espresso-in",
        pseudopotentials=pseudopotentials,
        ppdir=PSEUDOPOTENTIALS_DIR,
        input_data={"control": control, "system": system, "electrons": electrons},
        kpts=kpts or (4, 4, 4),  # default k-point grid for SCF
    )
    print(
        f"info: generated SCF input file at {os.path.relpath(input_file, os.getcwd())}"
    )
    # ---- handle NSCF input generation ----
    control.update(
        {
            "calculation": "nscf",
        }
    )
    input_file = os.path.join(out_dir, "nscf.in")
    write(
        input_file,
        atoms,
        format="espresso-in",
        pseudopotentials=pseudopotentials,
        ppdir=PSEUDOPOTENTIALS_DIR,
        input_data={"control": control, "system": system, "electrons": electrons},
        kpts=kpts or (4, 4, 4),  # denser k-point grid for NSCF
    )
    print(
        f"info: generated NSCF input file at {os.path.relpath(input_file, os.getcwd())}"
    )
    # ---- handle relaxation input generation ----
    control.update(
        {
            "calculation": "relax",  # change calculation type to relaxation
        }
    )
    input_file = os.path.join(out_dir, "relax.in")
    write(
        input_file,
        atoms,
        format="espresso-in",
        pseudopotentials=pseudopotentials,
        ppdir=PSEUDOPOTENTIALS_DIR,
        input_data={
            "control": control,
            "system": system,
            "electrons": electrons,
            "ions": {"ion_dynamics": "bfgs"},
        },
        kpts=kpts or (2, 2, 2),  # same k-point grid for relaxation
    )
    print(
        f"info: generated relaxation input file at {os.path.relpath(input_file, os.getcwd())}"
    )

    # ---- handle DOS input generation ----
    dos_data: QEInputType = {
        "prefix": prefix,
        "outdir": out_dir,
        "fildos": os.path.join(out_dir, "dos.result"),
        "emin": -10,
        "emax": 10,
        "DeltaE": 0.01,
    }
    dos_str = f"&DOS\n{dict_to_str(dos_data)}\n/\n"
    input_file = os.path.join(out_dir, "dos.in")
    with open(input_file, "w") as f:
        f.write(dos_str)
    print(
        f"info: generated DOS input file at {os.path.relpath(input_file, os.getcwd())}"
    )

    # ---- handle phonon input generation ----
    phonon_data: QEInputType = {
        "prefix": prefix,
        "outdir": out_dir,
        "fildyn": os.path.join(out_dir, "ph.result"),
        "tr2_ph": 1e-14,  # convergence threshold for phonon calculations
        "nq1": 4,  # number of q-points in the first direction
        "nq2": 4,  # number of q-points in the second direction
        "nq3": 4,  # number of q-points in the third direction
    }
    if like == "insulator":
        phonon_data.update(
            {
                "epsil": True,  # calculate dielectric properties
                "ldisp": True,
            }
        )
    phonon_str = f"&inputph\n{dict_to_str(phonon_data)}\n/\n"
    input_file = os.path.join(out_dir, "ph.in")
    with open(input_file, "w") as f:
        f.write(phonon_str)
    print(
        f"info: generated phonon input file at {os.path.relpath(input_file, os.getcwd())}"
    )


def dict_to_str(input_dict: dict) -> str:
    """convert a dictionary to a string representation for QE input files."""
    return "\n".join(
        [
            f"{key} = '{value}'"
            if isinstance(value, str)
            else f"{key} = .{str(value).lower()}."
            if isinstance(value, bool)
            else f"{key} = {value}"
            for key, value in input_dict.items()
        ]
    )

def main():
    # generate output directory
    project_dir = setup_output_project(PROJECT_NAME)

    # build your structure here
    structure = get_structure_from_id("mp-149")
    print_structure([structure])
    structure.to(filename=os.path.join(project_dir, "supercell.cif"), fmt="cif")

    # generate input files for Quantum ESPRESSO
    generate_espresso_input(
        structure,
        out_dir=project_dir,
        prefix="si",
        like="insulator",
        xc=FUNCTIONAL, # i extract the functional from the project name since my project name is in the format "project_functional" but you can change it to whatever you want
        kpts=(7, 7, 7),
    )


main()

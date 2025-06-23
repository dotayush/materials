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

import math
import os


import matplotlib
import numpy as np
import pandas as pd
from itertools import product
from typing import Union
from ase.atoms import Atoms
from ase.calculators.espresso import EspressoProfile
from ase.io import write
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core import Site, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


# matplotlib backend for plotting
matplotlib.use("Qt5Agg")

# load environment variables
_ = load_dotenv()
MP_API_KEY = os.getenv("MP_API_KEY")
if MP_API_KEY:
    mpr = MPRester(MP_API_KEY)

# directories for saved structures and pseudopotentials
SAVED_STRUCTURES_DIR = os.path.join(
    os.getcwd(), os.getenv("SAVED_STRUCTURES_DIR", "./saved_structures")
)
PSEUDOPOTENTIALS_DIR = os.path.join(os.getcwd(), os.getenv("PSEUDOS_DIR", "./pseudos"))
OUTPUT_DIR = os.path.join(os.getcwd(), os.getenv("OUTPUT_DIR", "out"))
PROJECT_NAME = os.getenv("PROJECT_NAME", "default_project")

espresso_profile = EspressoProfile(
    command="mpirun -np 4 pw.x", pseudo_dir=PSEUDOPOTENTIALS_DIR
)

QEInputType = dict[str, Union[str, float, int]]


def setup_output_project(project_name: str) -> str:
    """Setup the output directory for the project. If the directory already exists and has contents, it will error."""
    project_dir = os.path.join(os.getcwd(), OUTPUT_DIR, project_name)
    print(f"info: setting up project directory: {project_dir}")

    # error if the project directory already exists and has contents
    if os.path.exists(project_dir) and len(os.listdir(project_dir)) > 0:
        raise FileExistsError(
            f"error: project directory {project_dir} already exists and has contents. please remove the directory or choose a different project name."
        )

    os.makedirs(project_dir, exist_ok=True)
    return project_dir


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


def substrate_compatibility(structure_a: Structure, structure_b: Structure):
    """Calculate the match of two structures as a substrate and film pair."""
    analyzer = SubstrateAnalyzer()
    matches = analyzer.calculate(structure_a, structure_b)
    if not matches:
        print("info: no suitable substrate matches found")
        return None

    # return the match with the lowest von mises strain
    return min(matches, key=lambda x: x.strain.von_mises_strain)

def find_optimal_scale(required_scale: float, max_scale: int = 10):
    """Find scale factors (i, j, k) such that:
    - i * j * k >= required_scale
    - the product is as close as possible to required_scale
    - i, j, k are as balanced (close to each other) as possible
    """
    best_combo = (1, 1, 1)
    best_score = float('inf')

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

def fix_labels(structure: Structure):
    """Fix the labels of the sites in a structure to ensure they are unique."""
    labelled_sites = []
    label_counters = {}
    for site in structure:
        label = site.species_string
        if label not in label_counters:
            label_counters[label] = 0
        else:
            label_counters[label] += 1
            label = f"{label}_{label_counters[label]}"
        labelled_sites.append(Site(site.species, site.coords, label=label))

    labelled_supercell = Structure(
        lattice=structure.lattice,
        species=[site.species for site in labelled_sites],
        coords=[site.coords for site in labelled_sites],
        site_properties={},
    )
    return labelled_supercell

def create_doped_structure(
    structure: Structure,
    host_element: str,
    dopant_element: str,
    dopant_concentration: float,
):
    conventional = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    c_nat = len([site for site in conventional if site.species_string == host_element]) # 8 Si atoms
    t_nat = c_nat # let t_nat start from the conventional atoms = 8
    while True:
        if t_nat * dopant_concentration < 1: # if 8 * (1.1 / 100) <= 1 => 0.088 < 1, then t_nat++
            t_nat += 1
        else: # if 91 * (0.1 / 100) >= 1 => 1.001 >= 1, then we can stop; here t_nat = 91
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

    # fix labels
    labelled_supercell = fix_labels(supercell)

    data_str = f"{scale}({scale[0] * scale[1] * scale[2]})_{t_nat}/{c_nat}_{required_scale:.3f}_{len(supercell)}_{(1 / s_nat)*100:.3f}"

    return labelled_supercell, data_str


def generate_espresso_input(
    struct: Structure, out_dir: str, prefix: str, kpts=None, nbnd=None
):
    """Generate Quantum ESPRESSO input files for a given structure.
    This includes SCF, NSCF, and DOS input files."""
    atoms = AseAtomsAdaptor.get_atoms(struct)
    if not isinstance(atoms, Atoms):
        raise ValueError(
            f"error: expected a single Atoms object, but got {type(atoms)}"
        )
    atoms.wrap()

    # this is why we rename the pseudopotentials to lowercase in the first place with rename_psuedos.py
    pseudopotentials = {
        s: f"{s.lower()}.UPF" for s in set(atoms.get_chemical_symbols())
    }
    nat = len(atoms)

    # ---- handle SCF input generation ----
    control: QEInputType = {
        "calculation": "scf",  # default to SCF calculation
        "prefix": prefix,  # use the base name of the cif file as prefivx
        "pseudo_dir": PSEUDOPOTENTIALS_DIR,
        "outdir": out_dir,
        "verbosity": "high",
        "tstress": True,
        "tprnfor": True,
    }
    system: QEInputType = {
        "nat": nat,
        "ecutwfc": 30,
        "ecutrho": 120,
        "occupations": "smearing",  # default to smearing for SCF
        "smearing": "marzari-vanderbilt",  # default smearing type
        "degauss": 0.01,  # default smearing width
    }
    electrons: QEInputType = {
        "conv_thr": 1e-8,
        "mixing_beta": 0.4,
    }
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
    system.update(
        {
            "nbnd": nbnd or 4 * nat,  # number of bands, default to 4 * nat
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
        kpts=kpts or (8, 8, 8),  # denser k-point grid for NSCF
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
    # build dos string
    dos_str = "\n".join(
        [
            f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value:.6f}"
            for key, value in dos_data.items()
        ]
    )
    dos_str = f"&DOS\n{dos_str}\n/\n"
    input_file = os.path.join(out_dir, "dos.in")
    with open(input_file, "w") as f:
        f.write(dos_str)
    print(
        f"info: Generated DOS input file at {os.path.relpath(input_file, os.getcwd())}"
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
    # build phonon string
    phonon_str = "\n".join(
        [
            f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}"
            for key, value in phonon_data.items()
        ]
    )
    phonon_str = f"&inputph\n{phonon_str}\nldisp = .true.\n/\n"
    input_file = os.path.join(out_dir, "ph.in")
    with open(input_file, "w") as f:
        f.write(phonon_str)
    print(
        f"info: Generated phonon input file at {os.path.relpath(input_file, os.getcwd())}"
    )


def main():
    # setup project directory
    project_dir = setup_output_project(PROJECT_NAME)

    structure = SpacegroupAnalyzer(get_structure_from_id("mp-149")).get_conventional_standard_structure()
    structure.make_supercell(
         np.diag([2, 1, 1])
    )  # make a 2x1x1 supercell of Si i.e. # 16 Si atoms
    # structure.replace(0, "P")  # replace the first Si atom with P; this means 1/16 P doping or 6.25% P doping in Si crystal
    # supercell = fix_labels(structure)  # fix labels to ensure uniqueness; culprit for distorting the atom position in the supercell
    print_structure([structure])
    structure.to(filename=os.path.join(project_dir, "supercell.cif"), fmt="cif")

    # generate the input files for Quantum ESPRESSO
    generate_espresso_input(
        structure,
        out_dir=project_dir,
        prefix="si",
    )


if __name__ == "__main__":
    main()

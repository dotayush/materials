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
OUTPUT_DIR = os.path.join(os.getcwd(), os.getenv("OUTPUT_DIR", "./out"))

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


def get_anisotropic_scaling(n, concentration):
    target = 1 / concentration
    factors = []

    # Find minimal a×b×c where (n*a*b*c) % (1/concentration) == 0
    for v in range(1, 100):
        if (n * v) % target == 0:
            # Factorize v into a,b,c
            for a in range(1, v+1):
                if v % a != 0: continue
                for b in range(1, v//a+1):
                    c = v // (a*b)
                    if a*b*c == v:
                        factors.append((a,b,c))
            # Return most cubic-like factors
            return min(factors, key=lambda x: max(x)/min(x))

    return (1,1,math.ceil(target/n))  # Fallback


def create_doped_structure(
    structure: Structure,
    dopant_element: str,
    dopant_concentration: float,
):
    # get supercell size
    conventional = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    a_s, b_s, c_s = get_anisotropic_scaling(conventional, dopant_concentration)

    # make supercell
    supercell = conventional.make_supercell(np.diag([a_s, b_s, c_s]))
    t_nat = len(supercell)
    d_nat = int(round(dopant_concentration * t_nat))

    # calculate exact number of dopants

    # random substitution
    host_indices = [i for i, _ in enumerate(supercell)]
    np.random.shuffle(host_indices)
    for idx in host_indices[:d_nat]:
        supercell.replace(idx, dopant_element)

    # wrap the supercell to ensure all sites are within the unit cell
    # supercell = supercell.get_sorted_structure()
    # wrapped_supercell = supercell.copy()
    # wrapped_supercell = wrapped_supercell.make_supercell(
    #     np.diag([1, 1, 1]),
    # )

    # fix labels
    labelled_sites = []
    label_counters = {}
    for site in supercell:
        label = site.species_string
        if label not in label_counters:
            label_counters[label] = 0
        else:
            label_counters[label] += 1
            label = f"{label}_{label_counters[label]}"
        labelled_sites.append(Site(site.species, site.coords, label=label))

    labelled_supercell = Structure(
        lattice=supercell.lattice,
        species=[site.species for site in labelled_sites],
        coords=[site.coords for site in labelled_sites],
        site_properties={},
    )

    print(
        f"info: created {a_s}x{b_s}x{c_s} supercell: {t_nat} atoms (expected: {dopant_concentration * 100:.1f}% concentration). "
        f"calculated {(d_nat / len(labelled_supercell)) * 100:.1f}% doping concentration."
    )

    data_str = f"({a_s:.1f}x{b_s:.1f}x{c_s:.1f})_{d_nat}/{len(labelled_supercell)}_{(d_nat / len(labelled_supercell)) * 100:.1f}%"

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
    }
    system: QEInputType = {
        "nat": nat,
        "ecutwfc": 30,
        "ecutrho": 240,
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
        kpts=kpts or (4, 4, 4),  # same k-point grid for relaxation
    )
    print(
        f"info: generated relaxation input file at {os.path.relpath(input_file, os.getcwd())}"
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
    PROJECT_NAME = "si_p_doping"
    project_dir = setup_output_project(PROJECT_NAME)

    # create a doped structure with Si and P
    size_table: dict[str, str] = {}
    structure = get_structure_from_id("mp-149")
    for i in np.arange(1, 10.0, 1):
        print(f"info: generating doped structure with {i:.1f}% P doping")
        si_p, data_str = create_doped_structure(structure, "P", (i / 100.0))
        i_to_formatted_str = f"{i:.1f}".replace(".", "_")
        size_table[i_to_formatted_str] = data_str

    print(size_table)

    # structure = get_structure_from_id("mp-149")
    # si_p, _ = create_doped_structure(
    #     structure, "Si", "P", (6.2 / 100)
    # )
    # si_p.to(filename=os.path.join(project_dir, "supercell.cif"), fmt="cif")

    # # generate the input files for Quantum ESPRESSO
    # generate_espresso_input(
    #     si_p,
    #     out_dir=project_dir,
    #     prefix="si_p",
    # )


if __name__ == "__main__":
    main()

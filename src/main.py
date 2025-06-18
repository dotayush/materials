"""
    "Materials" is a collection of post-processing scripts to process and
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
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read, write
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.pwscf import PWOutput
from pymatgen.io.ase import AseAtomsAdaptor

_ = load_dotenv()
MP_API_KEY = os.getenv("MP_API_KEY")
mpr = MPRester(MP_API_KEY)
matplotlib.use("Qt5Agg")
SAVED_STRUCTURES_DIR = os.path.join(os.getcwd(), "saved_structures")
PSUEDOPOTENTIALS_DIR = os.path.join(os.getcwd(), "pseudos")
profile = EspressoProfile(command="mpirun -np 4 pw.x", pseudo_dir=PSUEDOPOTENTIALS_DIR)
QEInputType = dict[str, Union[str, float, int]]


def setup_output_project(project_name: str) -> str:
    """Setup the output directory for the project."""
    project_dir = os.path.join(os.getcwd(), "out", project_name)
    print(f"[Info] Setting up project directory: {project_dir}")
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def print_structure(structures, title=None):
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


def get_strucuture_from_id(mp_id: str) -> Structure:
    """Retrieve a structure from the Materials Project database using its ID."""
    # check under ./saved_structures if the structure is already saved
    structure_file = f"{mp_id}.cif"
    structure_path = os.path.join(SAVED_STRUCTURES_DIR, structure_file)
    print(f"[Info] looking for structure {mp_id} under {SAVED_STRUCTURES_DIR}...")
    if os.path.exists(structure_path):
        print(f"[Info] structure {mp_id} found in saved structures.")
        return Structure.from_file(structure_path)
    else:
        print(
            f"[Info] structure {mp_id} not found in saved structures. fetching from Materials Project..."
        )
        structure = mpr.get_structure_by_material_id(mp_id)
        # save the structure to the ./saved_structures directory
        os.makedirs(SAVED_STRUCTURES_DIR, exist_ok=True)
        if isinstance(
            structure, Structure
        ):  # should be the case since we never pass a list of ids
            print(f"Saving structure {mp_id} to {structure_path}")
            _ = structure.to_file(filename=structure_path, fmt="cif")
            return structure
        else:
            # well, this should never happen, but just in case
            raise ValueError(
                f"Expected a single Structure object, but got {type(structure)}"
            )


def substrate_compatibility(structure_a: Structure, structure_b: Structure):
    """Calculate the match of two structures as a substrate and film pair."""
    analyzer = SubstrateAnalyzer()
    matches = analyzer.calculate(structure_a, structure_b)
    if not matches:
        print("No suitable substrate matches found")
        return None

    # return the match with the lowest von Mises strain
    return min(matches, key=lambda x: x.strain.von_mises_strain)


def create_doped_structure(
    structure: Structure,
    host_element,
    dopant_element,
    dopant_concentration, # 0.0 ~ 1.0 (0.02 = 2% concentration etc.)
):
    """Create doped structure with specified concentration in a supercell"""
    num_host = sum(
        1 for site in structure if site.species_string == host_element
    )  # how many host atoms?
    scaling = max(
        1, math.ceil(int(math.cbrt(num_host/dopant_concentration)))
    )  # how much scale to have at least min_atoms of host_element
    supercell = structure * [
        scaling,
        scaling,
        scaling,
    ]  # create a supercell that has at least min_atoms of host_element

    host_sites = [
        i for i, site in enumerate(supercell) if site.species_string == host_element
    ]  # find the indices of the host_element in the supercell
    n_dopants = int(
        len(host_sites) * dopant_concentration
    )  # calculate how many dopants we want to introduce based on the concentration
    if n_dopants < 1:
        n_dopants = 1

    # Random substitution
    np.random.shuffle(
        host_sites
    )  # shuffle the indices to randomly select which host sites to replace
    for idx in host_sites[
        :n_dopants
    ]:  # replace the first n_dopants host sites with the dopant_element
        supercell.replace(idx, dopant_element)

    print(
        f"[Info] The strcture has {num_host} {host_element} atoms. It's scaled by {scaling}. The supercell ({supercell.formula}) is {scaling}x{scaling}x{scaling} or {len(supercell)} atoms. The concentration of {dopant_element} will be {dopant_concentration * 100:.2f}%."
    )

    return supercell


def generate_espresso_input(
    struct: Structure, out_dir: str, prefix: str, kpts=None, nbnd=None
):
    """Generate Quantum ESPRESSO input files for a given structure.
    This includes SCF, NSCF, and DOS input files."""
    atoms = AseAtomsAdaptor.get_atoms(struct)
    if not isinstance(atoms, Atoms):
        raise ValueError(f"Expected a single Atoms object, but got {type(atoms)}")
    pseudopotentials = {
        s: f"{s.lower()}.UPF" for s in set(atoms.get_chemical_symbols())
    }
    nat = len(atoms)

    # ---- handle SCF input generation ----
    control: QEInputType = {
        "calculation": "scf",  # default to SCF calculation
        "prefix": prefix,  # use the base name of the cif file as prefivx
        "pseudo_dir": PSUEDOPOTENTIALS_DIR,
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
        ppdir=PSUEDOPOTENTIALS_DIR,
        input_data={"control": control, "system": system, "electrons": electrons},
        kpts=kpts or (4, 4, 4),  # default k-point grid for SCF
    )
    print(f"[Info] Generated SCF input file at {input_file}")

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
        ppdir=PSUEDOPOTENTIALS_DIR,
        input_data={
            "control": control,
            "system": system,
            "electrons": electrons,
            "ions": {"ion_dynamics": "bfgs"},
        },
        kpts=kpts or (4, 4, 4),  # same k-point grid for relaxation
    )
    print(f"[Info] Generated relaxation input file at {input_file}")

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
        ppdir=PSUEDOPOTENTIALS_DIR,
        input_data={"control": control, "system": system, "electrons": electrons},
        kpts=kpts or (8, 8, 8),  # denser k-point grid for NSCF
    )
    print(f"[Info] Generated NSCF input file at {input_file}")

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
    print(f"[Info] Generated DOS input file at {input_file}")

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
    print(f"[Info] Generated phonon input file at {input_file}")


def generate_doped_si_with_p(project_dir: str, concentration=0.01):
    save_path = os.path.join(project_dir, "supercell.cif")
    if not os.path.exists(save_path):
        si_p = create_doped_structure(
            get_strucuture_from_id("mp-149"), "Si", "P", concentration
        )
        si_p.to(filename=save_path, fmt="cif")
        print(f"[Info] Saved doped Si structure to {save_path}")
    else:
        si_p = Structure.from_file(save_path)
        print(f"[Info] Loaded existing doped Si structure from {save_path}")
    return {"structure": si_p, "save_path": save_path}


def main():
    project_dir = setup_output_project("mp-149_si_p_doping")
    si_p = generate_doped_si_with_p(
        project_dir, concentration=(10 / 100)
    )  # doped Si with P @ 50% concentration

    # check if in files are already generated
    if not os.path.exists(os.path.join(project_dir, "scf.in")):
        generate_espresso_input(
            si_p["structure"],
            out_dir=project_dir,
            prefix="si_p",
        )
    else:
        pw_output = PWOutput(os.path.join(project_dir, "scf.out"))
        print(f"[Info] SCF output already exists at {pw_output.filename}")
        print(f"\nFinal Energy: {pw_output.final_energy} eV")
        print(f"Data: {pw_output.data}\n")


if __name__ == "__main__":
    main()

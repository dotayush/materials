import os
import matplotlib
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.io import read, write
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_ = load_dotenv()
MP_API_KEY = os.getenv("MP_API_KEY")
mpr = MPRester(MP_API_KEY)
matplotlib.use("Qt5Agg")
SAVED_STRUCTURES_DIR = os.path.join(os.getcwd(), "saved_structures")
PSUEDOPOTENTIALS_DIR = os.path.join(os.getcwd(), "pseudos")


def setup_output_project(project_name: str) -> str:
    """Setup the output directory for the project."""
    project_dir = os.path.join(os.getcwd(), "out", project_name)
    print(f"[Info] Setting up project directory: {project_dir}")
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


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
    dopant_concentration,
    min_atoms=1,
):
    """Create doped structure with specified concentration in a supercell"""
    num_host = sum(
        1 for site in structure if site.species_string == host_element
    )  # how many host atoms are in the structure that we want to substitute/dope with?
    scaling = max(
        1, int(np.ceil(min_atoms / num_host))
    )  # how many times do we need to scale the structure to have at least min_atoms of host_element?
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
    return supercell  # return the doped supercell structure


def generate_espresso_input(cif_file_path: str, calculation_type: str = "scf"):
    """Convert a pymatgen cif file to a Quantum ESPRESSO input file."""
    atoms = read(cif_file_path)
    if not isinstance(atoms, Atoms):
        raise ValueError(
            f"Expected an ASE Atoms object, but got {type(atoms)}. "
            "Ensure the CIF file is correctly formatted."
        )
    parent_path = os.path.dirname(cif_file_path)

    # generate the pseudopotentials dictionary
    pseudopotentials = {}
    for symbol in set(atoms.get_chemical_symbols()):
        pseudopotential_file = f"{symbol}.UPF"  # default pseudopotential file name
        pseudopotentials[symbol] = pseudopotential_file

    print(f"[Info] Using pseudopotentials: {pseudopotentials}")

    # generate the Quantum ESPRESSO input
    input_data = {
        "control": {
            "calculation": calculation_type,
            "prefix": "si_p",
            "pseudo_dir": PSUEDOPOTENTIALS_DIR,
            "outdir": parent_path,
            "verbosity": "high",
        },
        "system": {
            "ecutwfc": 30,
            "ecutrho": 240,
            "occupations": "smearing",
            "smearing": "gaussian",
            "degauss": 0.01,
            "nat": len(atoms),
        },
        "electrons": {
            "conv_thr": 1e-8,  # convergence threshold
            "mixing_beta": 0.4,  # mixing parameter for self-consistent field
            "electron_maxstep": 80,
        },
    }
    print(f"[Info] Using input data: {input_data}")

    calc = Espresso(input_data=input_data, pseudopotentials=pseudopotentials)
    atoms.set_calculator(calc)
    write(
        os.path.join(parent_path, "espresso_input.in"),
        atoms,
        format="espresso-in",
    )

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


def main():
    project_dir = setup_output_project("mp-149_si_p_doping")
    si = get_strucuture_from_id("mp-149")
    print_structure([si], title="Structures of Si")

    # compute match
    # match = substrate_compatibility(gan, si)
    # if match:
    #     film_miller = " ".join(map(str, match.film_miller))
    #     substrate_miller = " ".join(map(str, match.substrate_miller))
    #     compatibility_struct = [{
    #         "name": format(
    #             "{} on {}".format(
    #                 gan.composition.formula,
    #                 si.composition.formula,
    #             )
    #         ),
    #         "film_formula": gan.composition.formula,
    #         "film_miller": film_miller,
    #         "substrate_formula": si.composition.formula,
    #         "substrate_miller": substrate_miller,
    #         "strain": match.von_mises_strain * 100,
    #         "elastic_energy": match.elastic_energy,
    #     }]
    #     print("--------------------")
    #     print(pd.DataFrame(compatibility_struct))
    #     print("--------------------")
    # else:
    #     print("--------------------")
    #     print("No match found.")
    #     print("--------------------")

    # create a doped structure
    si_p = create_doped_structure(si, "Si", "P", 0.2)  # 20% P doping concentration
    print_structure([si_p], title="Doped Si with P ({}% concentration)".format(20))

    # make QE input
    si_p.to(filename=os.path.join(project_dir, "supercell.cif"), fmt="cif")
    generate_espresso_input(os.path.join(project_dir, "supercell.cif"))


if __name__ == "__main__":
    main()

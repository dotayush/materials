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

import numpy as np
import re
import os
import matplotlib.pyplot as plt
from scipy.constants import k
from dotenv import load_dotenv

_ = load_dotenv()
PROJECT_NAME = os.getenv("PROJECT_NAME", "default_project")
OUTPUT_DIR = os.path.join(os.getcwd(), os.getenv("OUTPUT_DIR", "out"))

BOHR_TO_ANG = 0.529177249  # bohr radius in angstroms
AVAGADRO = 6.02214076e23  # Avogadro's number'
RY_TO_EV = 13.6057039763  # Rydberg to electron volts conversion factor
EV_TO_J = 1.602176634e-19  # J/eV
A_TO_M = 1e-10  # m/Å
REDUCED_PLANCKS_CONSTANT = 1.054571817e-34  # J*s
PLANCKS_CONSTANT = 6.62607015e-34  # J*s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
M_ELECTRON = 9.10938356e-31  # electron mass in kg
VBM_THRESHOLD = 0.99  # threshold for considering a band as valence band maximum
CBM_THRESHOLD = 0.01  # threshold for considering a band as conduction band minimum


def get_crystal_type(self, type: int = 0):
    crystal_types = {
        0: "free",
        1: "cubic (sc)",
        2: "cubic (fc)",
        3: "cubic (bcc)",
        -3: "cubic (bcc) alt. axis",
        4: "hexagonal/trigonal",
        5: "trigonal (rhombohedral)",
        6: "triagonal (rhombohedral) alt. axis",
        7: "tetragonal (base-centered)",
        8: "orthorhombic",
        9: "orthorhombic (base-centered)",
        -9: "orthorhombic (base-centered) alt. axis",
        10: "orthorhombic (face-centered)",
        11: "orthorhombic (body-centered)",
        12: "monoclinic axis b",
        -12: "monoclinic axis c",
        13: "monoclinic (base-centered) axis c",
        -13: "monoclinic (base-centered) axis b",
        14: "triclinic",
    }
    return crystal_types.get(type, "unknown")


def process_bands(kpts_count: int = 0, kohn_sham_count: int = 0, filepath: str = ""):
    kpts = np.zeros((kpts_count, 3))  # initialize k-points array
    bands = np.zeros((kpts_count, kohn_sham_count))  # initialize bands array
    occupations = np.zeros(
        (kpts_count, kohn_sham_count)
    )  # initialize occupations array

    # read lines
    lines = []
    lc = 0
    with open(filepath, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if re.search(r"End of (self-consistent|band structure) calculation", line):
            lc = i
            print(f"gen:info: found end of calculation at line {lc + 1}")

    # extract k-points, band energies and occupations for each k-point
    band_index = 0
    while lc < len(lines):
        if re.search(r"k\s*=.*bands", lines[lc]):
            line = re.sub(
                r"(?<=[\d])(-)", r" \1", lines[lc].strip()
            )  # replace negative sign with space before it for easier regex extraction
            kpt_match = re.search(
                r"k\s*=\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", line
            )
            if kpt_match:
                # extract k-point index
                if band_index >= kpts_count:
                    raise ValueError(
                        f"more k-points found than expected: {band_index} vs {kpts_count}"
                    )
                kpt = np.array(
                    [
                        float(kpt_match.group(1)),
                        float(kpt_match.group(2)),
                        float(kpt_match.group(3)),
                    ]
                )
                kpts[band_index] = kpt
                band_index += 1

                # extract band energies
                next_lc = lc + 2
                all_point_energies = np.zeros(kohn_sham_count)
                extracted_energies = 0
                while next_lc < len(lines):
                    energy_line = lines[next_lc].strip()
                    if energy_line == "":
                        break
                    # data format: "-6.2995  -4.6919  -3.4715  -3.4714  -3.4714  -2.3700  -1.7552  -1.7552"
                    processed_energies = np.array(
                        [float(e) for e in re.findall(r"([-+]?\d*\.?\d+)", energy_line)]
                    )
                    all_point_energies[
                        extracted_energies : extracted_energies
                        + len(processed_energies)
                    ] = processed_energies
                    extracted_energies += len(processed_energies)
                    next_lc += 1
                if extracted_energies != kohn_sham_count:
                    raise ValueError(
                        f"extracted {extracted_energies} band energies, expected {kohn_sham_count}"
                    )
                bands[band_index - 1] = all_point_energies

                # extract occupations
                next_lc += 1
                all_point_occupations = np.zeros(kohn_sham_count)
                extracted_occupations = 0
                while next_lc < len(lines):
                    occupation_line = lines[next_lc].strip()
                    if occupation_line == "":
                        break
                    processed_occupations = np.array(
                        [
                            float(o)
                            for o in re.findall(r"([-+]?\d*\.?\d+)", occupation_line)
                        ]
                    )
                    all_point_occupations[
                        extracted_occupations : extracted_occupations
                        + len(processed_occupations)
                    ] = processed_occupations
                    extracted_occupations += len(processed_occupations)
                    next_lc += 1
                if extracted_occupations != kohn_sham_count:
                    raise ValueError(
                        f"extracted {extracted_occupations} occupations, expected {kohn_sham_count}"
                    )
                occupations[band_index - 1] = all_point_occupations
        lc += 1
    return kpts, bands, occupations


def process_gap(kpts, bands, occupations):
    max_local = np.zeros(kpts)
    min_local = np.zeros(kpts)

    for i in range(kpts):
        # find local maxima for valence band maximum
        vbm_candidates = bands[i, occupations[i] > VBM_THRESHOLD]
        if vbm_candidates.size > 0:
            max_local[i] = np.max(vbm_candidates)
        else:
            max_local[i] = -np.inf
        # find local minima for conduction band minimum
        cbm_candidates = bands[i, occupations[i] < CBM_THRESHOLD]
        if cbm_candidates.size > 0:
            min_local[i] = np.min(cbm_candidates)
        else:
            min_local[i] = np.inf

    vbm = np.max(max_local)
    cbm = np.min(min_local)
    bandgap = cbm - vbm

    return max_local, min_local, vbm, cbm, bandgap


def process_stress_tensor(filepath: str = ""):
    stress_tensor = np.zeros((3, 3))  # initialize stress tensor
    lines = []
    lc = 0
    with open(filepath, "r") as file:
        lines = file.readlines()
    while lc < len(lines):
        m = re.search(r"total\s+stress\s+\(Ry/bohr\*\*3\)", lines[lc])
        if m:
            s_lc = lc + 1
            for i in range(3):
                line = lines[s_lc].strip()
                if line == "":
                    s_lc += 1
                    continue
                # extract kBar stress components
                stress_components = re.findall(r"([-+]?\d*\.?\d+)", line)
                if len(stress_components) != 6:
                    raise ValueError(
                        f"expected 6 stress components, found {len(stress_components)} in line: {line}"
                    )
                for j in range(3):
                    # convert kBar to GPa (1 kBar = 0.1 GPa)
                    stress_tensor[i, j] = float(stress_components[j + 3]) * 0.1
                s_lc += 1
        lc += 1
    h_p = np.trace(stress_tensor) / 3

    von_mieses = np.sqrt(
        0.5
        * (
            (stress_tensor[0, 0] - stress_tensor[1, 1]) ** 2
            + (stress_tensor[1, 1] - stress_tensor[2, 2]) ** 2
            + (stress_tensor[2, 2] - stress_tensor[0, 0]) ** 2
            + 6
            * (
                stress_tensor[0, 1] ** 2
                + stress_tensor[1, 2] ** 2
                + stress_tensor[2, 0] ** 2
            )
        )
    )

    return (
        stress_tensor,
        h_p,
        von_mieses,
    )  # return stress tensor and hydrostatic pressure in GPa


def find_effective_mass(k_cart, energies, edge_idx):
    # get the two neighbors
    next_idx = (edge_idx + 1) if (edge_idx + 1 < len(energies)) else edge_idx
    prev_idx = (edge_idx - 1) if (edge_idx - 1 >= 0) else edge_idx

    print(f"gen:info: edge_idx={edge_idx}, next_idx={next_idx}, prev_idx={prev_idx}, len(energies)={len(energies)}")

    center_energy = energies[edge_idx]
    next_energy = energies[next_idx]
    prev_energy = energies[prev_idx]

    dk = np.linalg.norm(k_cart[next_idx] - k_cart[edge_idx])  * 1e10  # 1e10 for A -> m conversion
    d2E = (next_energy + prev_energy - 2 * center_energy) * EV_TO_J # EV_TO_J for eV -> J conversion

    print(f"gen:info: dk={dk}, d2E={d2E}, center_energy={center_energy}, next_energy={next_energy}, prev_energy={prev_energy}")

    # formulae: m_eff_kg = (hbar^2 * dk^2) / d2E
    m_eff_kg = abs((REDUCED_PLANCKS_CONSTANT**2 * dk**2) / d2E)
    print(f"gen:info: m_eff_kg={m_eff_kg}")

    # in m_e
    return m_eff_kg / M_ELECTRON


def intrinsic_carriers(cbm_energy_eV, vbm_energy_eV, m_eff_e, m_eff_h, T=300.0):
    BOLTZMANN_CONSTANT_EV = BOLTZMANN_CONSTANT / EV_TO_J  # J/K to eV/K
    Ei = 0.5 * (cbm_energy_eV + vbm_energy_eV) + (
        0.75 * BOLTZMANN_CONSTANT_EV * T * np.log(m_eff_h / m_eff_e)
    ) # intrinsic energy level in eV

    m_e = m_eff_e * M_ELECTRON # back to kg
    m_h = m_eff_h * M_ELECTRON # back to kg

    Nc = 2 * ((2 * np.pi * m_e * BOLTZMANN_CONSTANT * T) / (PLANCKS_CONSTANT**2)) ** 1.5
    Nv = 2 * ((2 * np.pi * m_h * BOLTZMANN_CONSTANT * T) / (PLANCKS_CONSTANT**2)) ** 1.5

    n = Nc * np.exp(-(cbm_energy_eV - Ei) * EV_TO_J / (BOLTZMANN_CONSTANT * T))
    p = Nv * np.exp(-(Ei - vbm_energy_eV) * EV_TO_J / (BOLTZMANN_CONSTANT * T))

    # convert to cm^-3
    return Ei, n / 1e6, p / 1e6


class SCF:
    def __init__(self, project_dir: str = ""):
        self.E_F = 0.0  # fermi energy in eV
        self.celldm = np.zeros(6)  # celldm parameters
        self.crystal_axis = np.zeros((3, 3))  # crystal axes in angstroms
        self.reciprocal_axis = np.zeros((3, 3))  # reciprocal crystal axes in angstrom
        self.alat = 0.0  # lattice parameter in bohr units
        self.crystal_index = 0  # bravais lattice index
        self.crystal_type = ""  # crystal type as a string
        self.volume = 0.0  # unit cell volume in angstrom^3
        self.total_atoms = 0  # total number of atoms in the unit cell
        self.total_atom_types = 0  # total number of atom types
        self.total_electrons = 0  # total number of electrons
        self.density = 0.0  # density in g/cm^3
        self.atom_types = {}  # dictionary of atom types and their masses
        self.total_energy = 0.0  # total energy in eV
        self.internal_energy = 0.0  # internal energy in eV
        self.kpts = 0  # number of k-points
        self.ks_states = 0  # number of kohn-sham states
        self.bands = np.zeros((0, 0))  # kohn-sham bands
        self.occupation = np.zeros((0, 0))  # kohn-sham occupation numbers
        self.kpoint_weights = np.zeros((0, 3))  # k-point weights
        self.max_local = np.zeros(0)  # local maxima for VBM
        self.min_local = np.zeros(0)  # local minima for CBM
        self.vbm = 0.0  # valence band maximum in eV
        self.cbm = 0.0  # conduction band minimum in eV
        self.bandgap = 0.0  # band gap energy in eV
        self.stress_tensor = np.zeros((3, 3))  # stress tensor in GPa
        self.hydrostatic_pressure = 0.0  # hydrostatic pressure in GPa
        self.von_mieses = 0.0  # von Mises stress in GPa
        self.file = os.path.relpath(os.path.join(project_dir, "scf.out"))
        self.constituents = {}  # what atom and how  much of it is present in the unit cell
        self.density = 0.0  # density in g/cm^3
        self.mass = 0.0  # total mass in grams
        if not os.path.exists(self.file):
            raise FileNotFoundError(
                f"scf:error: scf.out file was not found at address: {self.file}"
            )
        else:
            self.process_scf(filepath=self.file)

    def process_scf(self, filepath: str = ""):
        with open(filepath, "r") as file:
            scf_text = file.read()
            # fermi energy
            m = re.search(r"the Fermi energy is\s+([+\-\d\.]+)\s+ev", scf_text)
            if m:
                self.E_F = float(m.group(1))
            else:
                self.E_F = 0.0
                print("scf:warn: fermi energy not found.")

            # celldm
            m = re.findall(r"celldm\((\d)\)=\s+([-+]?[0-9]*\.?[0-9]+)", scf_text)
            if len(m) == 6:
                # celldm parameters found
                self.celldm = np.array([float(x[1]) for x in m])
            else:
                print("scf:warn: unit cell dimension (celldm) parameters not found.")

            # ibrav
            m = re.search(r"bravais-lattice index\s*=\s*(\d+)", scf_text)
            if m:
                self.crystal_index = int(m.group(1))
                self.crystal_type = f"{get_crystal_type(self.crystal_index)}"
            else:
                print(
                    "scf:warn: bravais lattice index (ibrav) not found, setting to 0 (free crystal)."
                )

            # lattice parameter (alat)  =      14.5482  a.u.
            m = re.search(
                r"lattice parameter \(alat\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*a\.u\.",
                scf_text,
            )
            if m:
                self.alat = float(m.group(1))
            else:
                print("scf:warn: lattice parameter (alat) not found")
                self.alat = 1.0
            self.alat *= BOHR_TO_ANG

            # crystal axis
            m = re.findall(
                r"a\((\d)\) = \(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                scf_text,
            )
            if m:
                for i in range(3):
                    for j in range(3):
                        self.crystal_axis[i, j] = float(m[i][j + 1])
            if self.crystal_index == 0:
                self.crystal_axis *= self.alat  # convert to angstroms

            # reciprocal axis
            m = re.findall(
                r"b\((\d)\) = \(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                scf_text,
            )
            if m:
                for i in range(3):
                    for j in range(3):
                        self.reciprocal_axis[i, j] = float(m[i][j + 1])
            if self.crystal_index == 0:
                self.reciprocal_axis *= (
                    2 * np.pi / self.alat
                )  # convert to reciprocal angstroms

            # cell volume
            m = re.search(r"unit-cell volume\s*=\s*([\d.]+)", scf_text)
            if m:
                self.volume = float(m.group(1)) * (BOHR_TO_ANG**3)
            else:
                print("scf:warn: unit-cell volume not found.")

            # total atoms
            m = re.search(r"number of atoms/cell\s*=\s*(\d+)", scf_text)
            if m:
                self.total_atoms = int(m.group(1))
            else:
                print("scf:warn: number of atoms in unit cell not found.")

            # total electrons
            m = re.search(r"number of electrons\s*=\s*([\d.]+)", scf_text)
            if m:
                self.total_electrons = int(float(m.group(1)))
            else:
                print("scf:warn: number of electrons not found.")

            # total atom types
            m = re.search(
                r"(?m)^ *atomic species +valence +mass +pseudopotential", scf_text
            )
            if m:
                start_idx = m.end()
                trimmed_text = scf_text[start_idx:].strip()
                pattern = re.compile(r"^\s*(\w+)\s+[\d.]+\s+([\d.]+)", re.MULTILINE)
                self.atom_types = {
                    m.group(1): float(m.group(2))
                    for m in pattern.finditer(trimmed_text)
                }
                self.total_atom_types = len(self.atom_types)

            # total energy
            m = re.search(
                r"^!\s*total energy\s*=\s*([-+]?\d*\.\d+|\d+)\s*Ry",
                scf_text,
                re.MULTILINE,
            )
            if m:
                self.total_energy = float(m.group(1)) * RY_TO_EV
            else:
                print("scf:warn: total energy not found.")

            # internal energy
            m = re.search(
                r"^ *internal energy E=F\+TS\s*=\s*([-+]?\d*\.\d+|\d+)\s*Ry",
                scf_text,
                re.MULTILINE,
            )
            if m:
                self.internal_energy = float(m.group(1)) * RY_TO_EV
            else:
                print("scf:warn: internal energy not found.")

            # k-points
            m = re.search(r"number of k points\s*=\s*(\d+)", scf_text)
            if m:
                self.kpts = int(m.group(1))
            else:
                print("scf:warn: number of k points not found.")

            # kohn-sham states
            m = re.search(r"number of Kohn-Sham states\s*=\s*([\d.]+)", scf_text)
            if m:
                self.ks_states = int(float(m.group(1)))
            else:
                print("scf:warn: number of Kohn-Sham states not found.")

            # energy bands and occupations
            self.bands, self.occupation = (
                np.zeros((self.kpts, self.ks_states)),
                np.zeros((self.kpts, self.ks_states)),
            )
            self.kpoint_weights, self.bands, self.occupation = process_bands(
                kpts_count=self.kpts,
                kohn_sham_count=self.ks_states,
                filepath=filepath,
            )
            (
                self.max_local,
                self.min_local,
                self.vbm,
                self.cbm,
                self.bandgap,
            ) = process_gap(self.kpts, self.bands, self.occupation)

            # stress tensor
            (
                self.stress_tensor,
                self.hydrostatic_pressure,
                self.von_mieses,
            ) = process_stress_tensor(filepath=filepath)

            # constituents in the unit cell
            m = re.search(
                r"(?m)^ *site n\.\s+atom\s+positions \(alat units\)", scf_text
            )
            if m:
                start_idx = m.end()
                end_idx = scf_text.find("\n\n", start_idx)
                trimmed_text = scf_text[start_idx:end_idx].strip()
                pattern = re.compile(
                    r"^\s*(\d+)\s+(\w+)\s+tau\(\s*\d+\)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)",
                    re.MULTILINE,
                )
                print(
                    f"scf:info: extracting constituents from scf.out file at {[m.group(2) for m in pattern.finditer(trimmed_text)]}"
                )
                for m in pattern.finditer(trimmed_text):
                    atom_type = m.group(2)
                    if atom_type not in self.constituents:
                        self.constituents[atom_type] = 0
                    self.constituents[atom_type] += 1
            else:
                print("scf:warn: constituents not found in scf.out file.")

            # density
            for atom, mass in self.atom_types.items():
                if atom not in self.constituents:
                    print(
                        f"scf:warn: atom {atom} not found in constituents but was found in atom types (check your scf.out file for data-errors). density calc. will not include this atom."
                    )
                    continue
                self.mass += mass * self.constituents[atom]
            self.mass /= AVAGADRO  # convert mass from g/mol to g
            self.density = self.mass / (
                self.volume * 1e-24
            )  # convert volume from angstrom^3 to cm^3

    def print_scf(self):
        print("-" * 30 + " SCF DATA " + "-" * 30)
        print(f"crystal index = {self.crystal_index}")
        print(f"crystal type = {self.crystal_type}")
        print(f"alat = {self.alat:.4f} angstrom)")
        print(f"fermi energy = {self.E_F} eV")
        print(f"celldm = {self.celldm.tolist()}")
        print("crystal axes (angstrom):")
        for i in range(3):
            print(f"\ta({i + 1}) = {self.crystal_axis[i].tolist()}")
        print("reciprocal crystal axes (1/angstorm):")
        for i in range(3):
            print(f"\tb({i + 1}) = {self.reciprocal_axis[i].tolist()}")
        print(f"total atoms in unit cell = {self.total_atoms}")
        print(f"total electrons in unit cell = {self.total_electrons}")
        print(f"total atom types in unit cell = {self.total_atom_types}")
        print("atom types:")
        for atom, mass in self.atom_types.items():
            print(f"\t{atom}: {mass} a.u.")
        print(f"total energy = {self.total_energy} eV")
        print(f"internal energy = {self.internal_energy} eV")
        print(f"number of k-points = {self.kpts}")
        print(f"number of kohn-sham states = {self.ks_states}")
        print(f"valence band maximum = {self.vbm} eV")
        print(f"conduction band minimum = {self.cbm} eV")
        print(f"band gap = {self.bandgap} eV")
        print("local maxima (VBM):")
        print(f"\t{self.max_local.tolist()}")
        print("local minima (CBM):")
        print(f"\t{self.min_local.tolist()}")
        print("stress tensor (GPa):")
        for i in range(3):
            print(f"\tx{i + 1} = {self.stress_tensor[i].tolist()}")
        print(f"hydrostatic/average pressure = {self.hydrostatic_pressure:.4f} GPa")
        print(f"von Mises stress = {self.von_mieses:.4f} GPa")
        print("constituents in the unit cell:")
        for atom, count in self.constituents.items():
            print(f"\t{atom}: {count} atoms")
        print(f"unit cell volume = {self.volume:.4f} angstrom^3")
        print(f"unit cell density = {self.density:.4f} g/cm^3")
        print(f"unit cell mass = {self.mass} g")
        print("region information:")
        print(
            "\tatoms/cm^3 = {:.2e} atoms/cm^3".format(
                self.total_atoms / (self.volume * 1e-24)
            )
        )  # convert volume from angstrom^3 to cm^3
        print(
            "\telectrons/cm^3 = {:.2e} electrons/cm^3".format(
                self.total_electrons / (self.volume * 1e-24)
            )
        )  # convert volume from angstrom^3 to cm^3
        print(
            "\tatoms/um^3 = {:.2e} atoms/um^3".format(
                self.total_atoms / (self.volume * 1e-12)
            )
        )  # convert volume from angstrom^3 to um^3
        print(
            "\telectrons/um^3 = {:.2e} electrons/um^3".format(
                self.total_electrons / (self.volume * 1e-12)
            )
        )  # convert volume from angstrom^3 to um^3
        print(
            "\tatoms/nm^3 = {:.2f} atoms/nm^3".format(
                self.total_atoms / (self.volume * 1e-3)
            )
        )  # convert volume from angstrom^3 to nm^3
        print(
            "\telectrons/nm^3 = {:.2f} electrons/nm^3".format(
                self.total_electrons / (self.volume * 1e-3)
            )
        )  # convert volume from angstrom^3 to nm^3


class NonSCF:
    def __init__(self, project_dir: str = ""):
        self.crystal_axis = np.zeros((3, 3))  # crystal axes in angstroms
        self.reciprocal_axis = np.zeros((3, 3))  # reciprocal crystal axes in angstrom
        self.alat = 0.0  # lattice parameter in bohr units
        self.crystal_index = 0  # bravais lattice index
        self.kpts = 0  # number of k-points
        self.ks_states = 0  # number of kohn-sham states
        self.bands = np.zeros((0, 0))  # kohn-sham bands
        self.occupation = np.zeros((0, 0))  # kohn-sham occupation numbers
        self.kpoint_weights = np.zeros((0, 3))  # k-point weights
        self.max_local = np.zeros(0)  # local maxima for VBM
        self.min_local = np.zeros(0)  # local minima for CBM
        self.vbm = 0.0  # valence band maximum in eV
        self.cbm = 0.0  # conduction band minimum in eV
        self.bandgap = 0.0  # band gap energy in eV
        self.file = os.path.relpath(os.path.join(project_dir, "nscf.out"))
        self.effective_mass_n = 0.0  # effective mass in m_e
        self.effective_mass_p = 0.0  # effective mass in m_e
        self.carrier_concentration_n = 0.0  # carrier concentration in cm^-3
        self.carrier_concentration_p = 0.0  # carrier concentration in cm^-3
        if not os.path.exists(self.file):
            raise FileNotFoundError(
                f"nscf:error: nscf.out file was not found at address: {self.file}"
            )
        else:
            self.process_nscf(filepath=self.file)

    def process_nscf(self, filepath: str = ""):
        with open(filepath, "r") as file:
            nscf_text = file.read()

            # ibrav; same as SCF, not rendered
            m = re.search(r"bravais-lattice index\s*=\s*(\d+)", nscf_text)
            if m:
                self.crystal_index = int(m.group(1))
                self.crystal_type = f"{get_crystal_type(self.crystal_index)}"
            else:
                print(
                    "scf:warn: bravais lattice index (ibrav) not found, setting to 0 (free crystal)."
                )
            # alat; same as SCF, not rendered
            m = re.search(
                r"lattice parameter \(alat\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*a\.u\.",
                nscf_text,
            )
            if m:
                self.alat = float(m.group(1))
            else:
                print("scf:warn: lattice parameter (alat) not found")
                self.alat = 1.0
            self.alat *= BOHR_TO_ANG  # convert to angstroms

            # crystal axis
            m = re.findall(
                r"a\((\d)\) = \(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                nscf_text,
            )
            if m:
                for i in range(3):
                    for j in range(3):
                        self.crystal_axis[i, j] = float(m[i][j + 1])
            self.crystal_axis *= BOHR_TO_ANG * self.alat  # convert to angstroms

            # reciprocal axis
            m = re.findall(
                r"b\((\d)\) = \(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                nscf_text,
            )
            if m:
                for i in range(3):
                    for j in range(3):
                        self.reciprocal_axis[i, j] = float(m[i][j + 1])
            self.reciprocal_axis *= BOHR_TO_ANG * (
                2 * np.pi / self.alat
            )  # convert to reciprocal angstroms

            # k-points
            m = re.search(r"number of k points\s*=\s*(\d+)", nscf_text)
            if m:
                self.kpts = int(m.group(1))
            else:
                print("nscf:warn: number of k points not found.")

            # kohn-sham states
            m = re.search(r"number of Kohn-Sham states\s*=\s*([\d.]+)", nscf_text)
            if m:
                self.ks_states = int(float(m.group(1)))
            else:
                print("nscf:warn: number of Kohn-Sham states not found.")

            # energy bands and occupations
            self.bands, self.occupation = (
                np.zeros((self.kpts, self.ks_states)),
                np.zeros((self.kpts, self.ks_states)),
            )
            self.kpoint_weights, self.bands, self.occupation = process_bands(
                self.kpts, self.ks_states, filepath
            )
            (
                self.max_local,
                self.min_local,
                self.vbm,
                self.cbm,
                self.bandgap,
            ) = process_gap(self.kpts, self.bands, self.occupation)

            # cbm = np.argmin(self.min_local)
            # dk = np.linalg.norm(k_cart[cbm + 1] - k_cart[cbm])
            # print("Δk between neighbor points (Å⁻¹):", dk)
            # effective mass
            k_cart = self.kpoint_weights @ self.reciprocal_axis
            self.effective_mass_n = find_effective_mass(
                k_cart,
                self.min_local,
                np.argmin(self.min_local),
            )
            self.effective_mass_p = find_effective_mass(
                k_cart,
                self.max_local,
                np.argmax(self.max_local),
            )

            # carrier concentration
            EV_i, self.carrier_concentration_n, self.carrier_concentration_p = (
                intrinsic_carriers(
                    cbm_energy_eV=self.cbm,
                    vbm_energy_eV=self.vbm,
                    m_eff_e=self.effective_mass_n,
                    m_eff_h=self.effective_mass_p,
                )
            )

    def print_nscf(self):
        print("-" * 30 + " NSCF DATA " + "-" * 30)
        print(f"number of k-points = {self.kpts}")
        print(f"number of kohn-sham states = {self.ks_states}")
        print(f"valence band maximum = {self.vbm} eV")
        print(f"conduction band minimum = {self.cbm} eV")
        print(f"band gap = {self.bandgap} eV")
        print("local maxima (VBM):")
        print(f"\t{self.max_local.tolist()}")
        print("local minima (CBM):")
        print(f"\t{self.min_local.tolist()}")
        print(f"effective mass (cbm/electrons) = {self.effective_mass_n} m_e")
        print(f"effective mass (vbm/holes) = {self.effective_mass_p} m_e")
        print(
            f"carrier concentration (cbm/electrons/n) = {self.carrier_concentration_n:.2e} cm^-3"
        )
        print(
            f"carrier concentration (vbm/holes/p) = {self.carrier_concentration_p:.2e} cm^-3"
        )


class Compound:
    def __init__(self, project_dir: str, scf=True, nscf=True):
        if not os.path.exists(project_dir):
            raise FileNotFoundError(
                f"compound:error project directory {project_dir} does not exist."
            )
        if scf and not os.path.exists(os.path.join(project_dir, "scf.out")):
            raise FileNotFoundError(
                f"compound:error scf.out file does not exist in {project_dir}."
            )
        else:
            self.scf = SCF(project_dir=project_dir)
        if nscf and not os.path.exists(os.path.join(project_dir, "nscf.out")):
            raise FileNotFoundError(
                f"compound:error nscf.out file does not exist in {project_dir}."
            )
        else:
            self.nscf = NonSCF(project_dir=project_dir)


def main():
    PROJECT_DIR = os.path.join(OUTPUT_DIR, PROJECT_NAME)
    compound = Compound(project_dir=PROJECT_DIR)
    compound.scf.print_scf()
    compound.nscf.print_nscf()


if __name__ == "__main__":
    main()

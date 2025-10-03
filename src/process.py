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

import json
from matplotlib.lines import lineStyles
import numpy as np
import re
import os
from shared import FUNCTIONAL, OUTPUT_DIR, PROJECT_NAME

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


def process_gap(kpoints, band_count, band_data, occupation_data):
    max_local = [(float("-inf"), -1)] * kpoints   # (value, band_index)
    min_local = [(float("inf"), -1)] * kpoints

    # this doesn't look right...
    # shape: (kpoint, band_energy_at_kpoint)
    # shape: (kpoint, band_occupation_at_kpoint)

    for i in range(kpoints):
        for j in range(band_count):
            if occupation_data[i, j] >= VBM_THRESHOLD:
                if band_data[i, j] > max_local[i][0]:
                    max_local[i] = (band_data[i, j], j)
            if occupation_data[i, j] <= CBM_THRESHOLD:
                if min_local[i][0] == 0.0 or band_data[i, j] < min_local[i][0]:
                    min_local[i] = (band_data[i, j], j)

    # 172 kpoints, 32 bands
    energies_only_vbm = [val[0] for val in max_local]
    energies_only_cbm = [val[0] for val in min_local]
    vbm = np.max(energies_only_vbm)
    cbm = np.min(energies_only_cbm)
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

def effective_mass_1d(k_cart, bands, band_index, edge_idx, window=0.2, direction=None):
    if direction is not None:
        # use projection if explicit direction is given (for 3D grids)
        direction = np.array(direction) / np.linalg.norm(direction)
        k_scalar = k_cart @ direction
    else:
        # assume bandstructure path -> use cumulative distance
        dk = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
        k_scalar = np.concatenate(([0], np.cumsum(dk)))

    # energies for the chosen band
    E_vals = bands[:, band_index]

    # shift reference to band edge
    k_shift = k_scalar - k_scalar[edge_idx]
    E_shift = E_vals - E_vals[edge_idx]

    # select only nearby points
    mask = np.abs(k_shift) < window
    if np.count_nonzero(mask) < 3:
        # fallback: always include 3 nearest neighbors
        nearest = np.argsort(np.abs(k_shift))[:5]  # edge + 4 neighbors
        mask = np.zeros_like(k_shift, dtype=bool)
        mask[nearest] = True

    k_fit = k_shift[mask]
    E_fit = E_shift[mask]

    # quadratic fit: E ≈ a k²
    coeffs = np.polyfit(k_fit, E_fit, 2)
    a = coeffs[0]
    a_si = a * EV_TO_J * (A_TO_M**2)  # convert eV·Å² → J·m²

    # m* = ħ² / (2a)
    m_kg = (REDUCED_PLANCKS_CONSTANT**2) / (2 * a_si)
    return abs(m_kg) / M_ELECTRON  # return in units of m_e


def intrinsic_carriers(effective_electron_mass, effective_hole_mass, bandgap, T=300.0):
    Nc = 2 * ((2 * np.pi *  (effective_electron_mass * M_ELECTRON) * BOLTZMANN_CONSTANT * T) / (PLANCKS_CONSTANT**2)) ** 1.5
    Nv = 2 * ((2 * np.pi * (effective_hole_mass * M_ELECTRON) * BOLTZMANN_CONSTANT * T) / (PLANCKS_CONSTANT**2)) ** 1.5
    ni = np.sqrt(Nc * Nv) * np.exp(-bandgap*EV_TO_J / (2*BOLTZMANN_CONSTANT*T))
    # convert to cm^-3
    return Nc * 1e-6, Nv * 1e-6, ni * 1e-6

def carrier_concentrations(E_F, vbm, cbm, nc, nv, T=300.0):
    n = (nc/1e-6) * np.exp((-(cbm - E_F) * EV_TO_J) / (BOLTZMANN_CONSTANT * T))
    p = (nv/1e-6) * np.exp((-(E_F - vbm) * EV_TO_J) / (BOLTZMANN_CONSTANT * T))
    return n * 1e-6, p * 1e-6  # convert to cm^-3



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
        self.max_local = np.empty(0, dtype=object)  # local maxima for VBM
        self.min_local = np.empty(0, dtype=object)  # local minima for CBM
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
        self.effective_electron_mass = 0.0  # effective mass in units of m_e
        self.effective_hole_mass = 0.0  # effective mass in units of m_e
        self.dos_nc = 0  # number of conduction band states in effective DOS calc (cm^-3)
        self.dos_nv = 0  # number of valence band states in effective DOS calc (cm^-3)
        self.electron_concentration = 0.0  # electron concentration (cm^-3)
        self.hole_concentration = 0.0  # hole concentration (cm^-3)
        self.instrinsic_concentration = 0.0  # carrier concentration (cm^-3)
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
            ) = process_gap(self.kpts, self.ks_states, self.bands, self.occupation)

            # if fermi energy was not found, set it to mid-gap
            if self.E_F == 0.0:
                if self.bandgap > 0:
                    self.E_F = (self.vbm + self.cbm) / 2

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

            k_cart = self.kpoint_weights @ self.reciprocal_axis
            self.effective_electron_mass = effective_mass_1d(
                k_cart=k_cart,
                bands=self.bands,
                band_index=np.argmin(self.min_local),
                edge_idx=np.argmin(self.min_local),
            )
            self.effective_hole_mass = effective_mass_1d(
                k_cart=k_cart,
                bands=self.bands,
                band_index=np.argmax(self.max_local),
                edge_idx=np.argmax(self.max_local),
            )

            # carrier concentration
            self.dos_nc, self.dos_nv, self.instrinsic_concentration = (
                intrinsic_carriers(
                    effective_electron_mass=self.effective_electron_mass,
                    effective_hole_mass=self.effective_hole_mass,
                    bandgap=self.bandgap,
                )
            )

            self.electron_concentration, self.hole_concentration = carrier_concentrations(
                E_F=self.E_F,
                vbm=self.vbm,
                cbm=self.cbm,
                nc=self.dos_nc,
                nv=self.dos_nv,
            )


class NonSCF:
    def __init__(self, project_dir: str = ""):
        self.E_F = 0.0  # fermi energy in eV
        self.crystal_axis = np.zeros((3, 3))  # crystal axes (angstrom)
        self.reciprocal_axis = np.zeros((3, 3))  # reciprocal crystal axes (angstrom)
        self.alat = 0.0  # lattice parameter (bohr units)
        self.crystal_index = 0  # bravais lattice index
        self.kpts = 0  # number of k-points
        self.ks_states = 0  # number of kohn-sham states
        self.bands = np.zeros((0, 0))  # kohn-sham bands
        self.occupation = np.zeros((0, 0))  # kohn-sham occupation numbers
        self.kpoint_weights = np.zeros((0, 3))  # k-point weights
        self.max_local = np.zeros(0)  # local maxima for VBM (eV[])
        self.min_local = np.zeros(0)  # local minima for CBM (eV[])
        self.vbm = 0.0  # valence band maximum (eV)
        self.cbm = 0.0  # conduction band minimum (eV)
        self.bandgap = 0.0  # band gap energy (eV)
        self.file = os.path.relpath(os.path.join(project_dir, "nscf.out"))
        self.effective_electron_mass = 0.0  # effective mass (m_e)
        self.effective_hole_mass = 0.0  # effective mass (m_e)
        self.dos_nc = 0  # number of conduction band states in effective DOS calc (cm^-3)
        self.dos_nv = 0  # number of valence band states in effective DOS calc (cm^-3)
        self.electron_concentration = 0.0  # electron concentration (cm^-3)
        self.hole_concentration = 0.0  # hole concentration (cm^-3)
        self.instrinsic_concentration = 0.0  # instrinsic concentration (cm^-3)
        if not os.path.exists(self.file):
            raise FileNotFoundError(
                f"nscf:error: nscf.out file was not found at address: {self.file}"
            )
        else:
            self.process_nscf(filepath=self.file)

    def process_nscf(self, filepath: str = ""):
        with open(filepath, "r") as file:
            nscf_text = file.read()

            # fermi energy
            m = re.search(r"the Fermi energy is\s+([+\-\d\.]+)\s+ev", nscf_text)
            if m:
                self.E_F = float(m.group(1))
            else:
                self.E_F = 0.0
                print("nscf:warn: fermi energy not found.")


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
            if self.crystal_index == 0:
                self.crystal_axis *= self.alat  # convert to angstroms

            # reciprocal axis
            m = re.findall(
                r"b\((\d)\) = \(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                nscf_text,
            )
            if m:
                for i in range(3):
                    for j in range(3):
                        self.reciprocal_axis[i, j] = float(m[i][j + 1])
            if self.crystal_index == 0:
                self.reciprocal_axis *= (
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
            ) = process_gap(self.kpts, self.ks_states, self.bands, self.occupation)

            # if fermi energy was not found, set it to mid-gap
            if self.E_F == 0.0:
                if self.bandgap > 0:
                    self.E_F = (self.vbm + self.cbm) / 2

            # cbm = np.argmin(self.min_local)
            # dk = np.linalg.norm(k_cart[cbm + 1] - k_cart[cbm])
            # print("Δk between neighbor points (Å⁻¹):", dk)
            # effective mass
            k_cart = self.kpoint_weights @ self.reciprocal_axis
            self.effective_electron_mass = effective_mass_1d(
                k_cart=k_cart,
                bands=self.bands,
                band_index=np.argmin(self.min_local),
                edge_idx=np.argmin(self.min_local),
            )
            self.effective_hole_mass = effective_mass_1d(
                k_cart=k_cart,
                bands=self.bands,
                band_index=np.argmax(self.max_local),
                edge_idx=np.argmax(self.max_local),
            )


            # instrinsic concentration
            self.dos_nc, self.dos_nv, self.instrinsic_concentration = (
                intrinsic_carriers(
                    effective_electron_mass=self.effective_electron_mass,
                    effective_hole_mass=self.effective_hole_mass,
                    bandgap=self.bandgap,
                )
            )

            # carrier concentration
            self.electron_concentration, self.hole_concentration = carrier_concentrations(
                E_F=self.E_F,
                vbm=self.vbm,
                cbm=self.cbm,
                nc=self.dos_nc,
                nv=self.dos_nv,
            )


class Dos:
    def __init__(self, project_dir: str = ""):
        self.file = os.path.relpath(os.path.join(project_dir, "dos.result"))
        #  E (eV)   dos(E)     Int dos(E) EFermi =    6.132 eV
        self.eneriges = np.zeros(0)  # energy values in eV
        self.dos = np.zeros(0)  # density of states values in states/eV
        self.int_dos = np.zeros(0)  # integrated density of states in states
        self.E_F = 0.0  # fermi energy in eV
        if not os.path.exists(self.file):
            raise FileNotFoundError(
                f"dos:error: dos.result file was not found at address: {self.file}"
            )
        else:
            self.process_dos(filepath=self.file)

    def process_dos(self, filepath: str = ""):
        energies = []
        dos_vals = []
        int_dos_vals = []
        with open(filepath, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() == "" or line.startswith("#"):
                    fermi_e = re.search(r"EFermi\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*eV", line)
                    if fermi_e:
                        self.E_F = float(fermi_e.group(1))
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    energies.append(float(parts[0]))
                    dos_vals.append(float(parts[1]))
                    int_dos_vals.append(float(parts[2]))
                except ValueError:
                    continue
        self.eneriges = np.array(energies)
        self.dos = np.array(dos_vals)
        self.int_dos = np.array(int_dos_vals)


class Compound:
    def __init__(self, scf=True, nscf=True):
        self.project_name = PROJECT_NAME
        self.functional = FUNCTIONAL
        self.project_dir = os.path.join(OUTPUT_DIR, self.project_name)
        self.scf = SCF(project_dir=self.project_dir)
        self.nscf = NonSCF(project_dir=self.project_dir)
        self.dos = Dos(project_dir=self.project_dir) if (scf and nscf) else None

    def present(self):
        return {
            "project": self.project_name,
            "xc_used": self.functional,
            "scf": {
                "E_F_(eV)": self.scf.E_F,
                "celldm": self.scf.celldm.tolist(),
                "crystal_axis_(angstrom)": self.scf.crystal_axis.tolist(),
                "reciprocal_axis_(1/angstrom)": self.scf.reciprocal_axis.tolist(),
                "alat_(angstrom)": self.scf.alat,
                "crystal_index": self.scf.crystal_index,
                "crystal_type": self.scf.crystal_type,
                "volume_(angstrom^3)": self.scf.volume,
                "total_atoms": self.scf.total_atoms,
                "total_electrons": self.scf.total_electrons,
                "total_elements": self.scf.total_atom_types,
                "total_kpoints": self.scf.kpts,
                "total_kohnsham_states": self.scf.ks_states,
                "density_(g/cm^3)": self.scf.density,
                "elements": self.scf.atom_types,
                "elements_count": self.scf.constituents,
                "total_energy_(eV)": self.scf.total_energy,
                "internal_energy_(eV)": self.scf.internal_energy,
                "vbm_(eV)": self.scf.vbm,
                "cbm_(eV)": self.scf.cbm,
                "bandgap_(eV)": self.scf.bandgap,
                "stress_tensor": self.scf.stress_tensor.tolist(),
                "hydrostatic_pressure_(GPa)": self.scf.hydrostatic_pressure,
                "von_mieses_stress_(GPa)": self.scf.von_mieses,
                "effective_electron_mass(me)": self.scf.effective_electron_mass,
                "effective_hole_mass(me)": self.scf.effective_hole_mass,
                "nc_dos_conduction_states(cm^-3)": self.scf.dos_nc,
                "nv_dos_valence_states(cm^-3)": self.scf.dos_nv,
                "electron_concentration(cm^-3)": self.scf.electron_concentration,
                "hole_concentration(cm^-3)": self.scf.hole_concentration,
                "intrinsic_concentration(cm^-3)": self.scf.instrinsic_concentration,
                "mass_(g)": self.scf.mass,
            },
            "nscf": {
                "E_F_(eV)": self.nscf.E_F,
                "vbm_(eV)": self.nscf.vbm,
                "cbm_(eV)": self.nscf.cbm,
                "bandgap_(eV)": self.nscf.bandgap,
                "total_kpoints": self.nscf.kpts,
                "total_kohnsham_states": self.nscf.ks_states,
                "effective_electron_mass(me)": self.nscf.effective_electron_mass,
                "effective_hole_mass(me)": self.nscf.effective_hole_mass,
                "nc_dos_conduction_states(cm^-3)": self.nscf.dos_nc,
                "nv_dos_valence_states(cm^-3)": self.nscf.dos_nv,
                "electron_concentration(cm^-3)": self.nscf.electron_concentration,
                "hole_concentration(cm^-3)": self.nscf.hole_concentration,
                "intrinsic_concentration(cm^-3)": self.nscf.instrinsic_concentration,
            },
        }

    def plot_min_max_local(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        # plot points
        if self.nscf:
            energies_only_vbm = [val[0] for val in self.scf.max_local]
            energies_only_cbm = [val[0] for val in self.scf.min_local]
            plt.plot(energies_only_vbm, label="VBM local maxima", color="blue", marker='x', linestyle='None')
            plt.plot(energies_only_cbm, label="CBM local minima", color="red", marker='o', linestyle='None')
            plt.scatter(np.argmax(energies_only_vbm), self.scf.vbm, color="blue", s=100, label="VBM")
            plt.scatter(np.argmin(energies_only_cbm), self.scf.cbm, color="red", s=100, label="CBM")
            plt.axhline(y=self.scf.E_F, color="green", linestyle="--", label="Fermi Level")
            # plot bands and wavevectors
            for band_idx in range(self.scf.ks_states):
                plt.plot(self.scf.bands[:, band_idx], color="black", alpha=0.5)
            for kpt_idx in range(self.scf.kpts):
                plt.axvline(x=kpt_idx, color="lightgray", alpha=0.5)

            plt.title(f"Band Structure Local Extrema for {self.project_name} ({self.scf.crystal_type})")
            plt.xlabel("K-point Index")
            plt.ylabel("Energy (eV)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
        else:
            print("gen:info no NSCF data available to plot.")

        plt.figure(figsize=(10, 6))
        # plot DOS
        if self.dos:
            plt.plot(self.dos.eneriges, self.dos.dos, color="purple", marker='x', linestyle='None', label="DOS")
            plt.plot(self.dos.eneriges, self.dos.dos, color="black", alpha=0.5)
            plt.axvline(x=self.scf.vbm, color="blue", linestyle="--", label="VBM")
            plt.axvline(x=self.scf.cbm, color="red", linestyle="--", label="CBM")
            plt.axvline(x=self.scf.E_F, color="green", linestyle="--", label="Fermi Level")
            plt.title(f"Density of States for {self.project_name} ({self.scf.crystal_type})")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Density of States (states/eV)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
        else:
            print("gen:info no DOS data available to plot.")

        plt.show()

    def save(self):
        output_file = os.path.join(self.project_dir, "processed.json")
        with open(output_file, "w") as f:
            json.dump(self.present(), f, indent=4, default =lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print(f"process:info: compound data saved to {output_file}")

compound = Compound()
print(json.dumps(compound.present(), indent=4))
compound.plot_min_max_local()

# Compound().save()

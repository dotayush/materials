import numpy as np
import re
import os
import matplotlib.pyplot as plt

PROJECT_DIR = "./out/mp-149_si_p_doping"
BOHR_TO_ANG = 0.529177249 # bohr radius in angstroms
AVAGADRO = 6.02214076e23  # Avogadro's number'
RY_TO_EV = 13.6057039763 # Rydberg to electron volts conversion factor
VBM_THRESHOLD = 0.99 # threshold for considering a band as valence band maximum
CBM_THRESHOLD = 0.01 # threshold for considering a band as conduction band minimum

class Compound:
    def __init__(self):
        self.E_F = 0.0
        self.celldm = np.zeros(6)
        self.crystal_axis = np.zeros((3, 3))  # 3x3 matrix for crystal axes
        self.alat = 0.0  # lattice parameter (scale) in bohr units
        self.crystal_index = 0
        self.crystal_type = ""
        self.volume = 0.0  # volume in angstrom^3
        self.total_atoms = 0  # total number of atoms in the unit cell
        self.total_atom_types = 0
        self.total_electrons = 0
        self.density = 0.0  # density in g/cm^3
        self.atom_types = {}
        self.total_energy = 0.0  # total energy in eV
        self.internal_energy = 0.0  # internal energy in eV
        self.kpts = 0  # number of k-points
        self.ks_states = 0  # number of kohn-sham states
        self.bands = np.zeros((0, 0))  # kohn-sham bands
        self.occupation = np.zeros((0, 0))  # kohn-sham occupation numbers
        self.kpoint_weights = np.zeros((0, 3))  # k-point weights
        self.max_local = np.zeros(0)
        self.min_local = np.zeros(0)
        self.vbm = 0.0  # valence band maximum
        self.cbm = 0.0  # conduction band minimum
        self.bandgap = 0.0  # band gap energy in eV

    def print(self):
        print("-" * 40)
        print(f"crystal index = {self.crystal_index}")
        print(f"crystal type = {self.crystal_type}")
        print(f"fermi energy = {self.E_F} eV")
        print(f"celldm = {self.celldm.tolist()}")
        print(f"alat = {self.alat} bohr")
        print("crystal axis (angstrom):")
        for i in range(3):
            print(f"\ta({i+1}) = {self.crystal_axis[i].tolist()}")
        print(f"unit cell volume = {self.volume} angstrom^3")
        print(f"total atoms = {self.total_atoms}")
        print(f"total electrons = {self.total_electrons}")
        print(f"total atom types = {self.total_atom_types}")
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

def main():
    compound = Compound()
    file_path = os.path.join(PROJECT_DIR, "scf.out")
    with open(file_path, "r") as f:
        scf_text = f.read()

        # fermi energy
        m = re.search(r"the Fermi energy is\s+([+\-\d\.]+)\s+ev", scf_text)
        if m:
            compound.E_F = float(m.group(1))
        else:
            compound.E_F = 0.0
            print("warn: fermi energy not found, setting to 0.0 eV")

        # celldm
        m = re.findall(r"celldm\((\d)\)=\s+([-+]?[0-9]*\.?[0-9]+)", scf_text)
        if len(m) != 6:
            raise ValueError("celldm not found.")
        compound.celldm = np.array([float(x[1]) for x in m])

        # ibrav
        m = re.search(r"bravais-lattice index\s*=\s*(\d+)", scf_text)
        if m is None:
            raise ValueError("crystal type not found.")
        ct = int(m.group(1))
        compound.crystal_type = f"{compound.get_crystal_type(ct)}"
        compound.crystal_index = ct

        # lattice parameter (alat)  =      14.5482  a.u.
        m = re.search(r"lattice parameter \(alat\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*a\.u\.", scf_text)
        if m is None:
            raise ValueError("lattice parameter (alat) not found.")
        compound.alat = float(m.group(1))

        # crystal axis
        m = re.findall(r"a\((\d)\) = \(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", scf_text)
        for i in range(3):
            for j in range(3):
                compound.crystal_axis[i, j] = float(m[i][j + 1])
        if ct == 0:
            # only convert to angstroms if free crystal
            compound.crystal_axis *= BOHR_TO_ANG * compound.alat  # convert to angstroms
        else:
            print("warn: only free crystal has crystal axis in angstroms, skipping axis conversion")


        # volume volume = np.abs(np.dot(a1, np.cross(a2, a3)))
        m = re.search(r"unit-cell volume\s*=\s*([\d.]+)", scf_text)
        if m is None:
            raise ValueError("unit-cell volume not found.")
        compound.volume = float(m.group(1)) * (BOHR_TO_ANG ** 3)

        # total atoms, electrons and atom types
        m = re.search(r"number of atoms/cell\s*=\s*(\d+)", scf_text)
        if m is None:
            raise ValueError("total atoms not found.")
        compound.total_atoms = int(m.group(1))
        m = re.search(r"number of electrons\s*=\s*([\d.]+)", scf_text)
        if m is None:
            raise ValueError("total electrons not found.")
        compound.total_electrons = int(float(m.group(1)))
        m = re.search(r"(?m)^ *atomic species +valence +mass +pseudopotential", scf_text)
        if m:
            start_idx = m.end()
            trimmed_text = scf_text[start_idx:].strip()
            pattern = re.compile(r'^\s*(\w+)\s+[\d.]+\s+([\d.]+)', re.MULTILINE)
            compound.atom_types = {m.group(1): float(m.group(2)) for m in pattern.finditer(trimmed_text)}
            compound.total_atom_types = len(compound.atom_types)

        # total energy
        m = re.search(r"^!\s*total energy\s*=\s*([-+]?\d*\.\d+|\d+)\s*Ry", scf_text, re.MULTILINE)
        if m:
            compound.total_energy = float(m.group(1)) * RY_TO_EV
        else:
            compound.total_energy = 0.0
            print("warn: total energy not found, setting to 0.0 eV")

        # internal energy
        m = re.search(r"^ *internal energy E=F\+TS\s*=\s*([-+]?\d*\.\d+|\d+)\s*Ry", scf_text, re.MULTILINE)
        if m:
            compound.internal_energy = float(m.group(1)) * RY_TO_EV
        else:
            compound.internal_energy = 0.0
            print("warn: internal energy not found, setting to 0.0 eV")

        # k-points
        m = re.search(r"number of k points\s*=\s*(\d+)", scf_text)
        if m is None:
            raise ValueError("number of k points not found.")
        compound.kpts = int(m.group(1))

        # kohn-sham states
        m = re.search(r"number of Kohn-Sham states\s*=\s*([\d.]+)", scf_text)
        if m is None:
            raise ValueError("kohn sham states not found.")
        compound.ks_states = int(float(m.group(1)))

        # bands
        compound.bands = np.zeros((compound.kpts, compound.ks_states))
        compound.occupation = np.zeros((compound.kpts, compound.ks_states))

        kpts, bands, occupations = process_bands(kpts_count = compound.kpts, kohn_sham_count = compound.ks_states, filepath = file_path)
        compound.kpoint_weights = kpts
        compound.bands = bands
        compound.occupation = occupations

        # vbm needs local maxima and cbm needs local minima
        compound.max_local = np.zeros(compound.kpts)
        compound.min_local = np.zeros(compound.kpts)

        for i in range(compound.kpts):
            # find local maxima for valence band maximum
            vbm_candidates = compound.bands[i, compound.occupation[i] > VBM_THRESHOLD]
            if vbm_candidates.size > 0:
                compound.max_local[i] = np.max(vbm_candidates)
            else:
                compound.max_local[i] = -np.inf
            cbm_candidates = compound.bands[i, compound.occupation[i] < CBM_THRESHOLD]
            if cbm_candidates.size > 0:
                compound.min_local[i] = np.min(cbm_candidates)
            else:
                compound.min_local[i] = np.inf

        compound.vbm = np.max(compound.max_local)
        compound.cbm = np.min(compound.min_local)
        compound.bandgap = compound.cbm - compound.vbm

        compound.print()

def process_bands(kpts_count: int = 0, kohn_sham_count: int = 0, filepath: str = ""):
    kpts = np.zeros((kpts_count, 3))  # initialize k-points array
    bands = np.zeros((kpts_count, kohn_sham_count))  # initialize bands array
    occupations = np.zeros((kpts_count, kohn_sham_count))  # initialize occupations array

    # read lines
    lines = []
    lc = 0
    with open(filepath, "r") as file:
        lines = file.readlines()

    # extract k-points, band energies and occupations for each k-point
    band_index = 0
    while lc < len(lines):
        if re.search(r"k\s*=.*bands", lines[lc]):
            line = re.sub(r"(?<=[\d])(-)", r" \1", lines[lc].strip())  # replace negative sign with space before it for easier regex extraction
            kpt_match = re.search(r"k\s*=\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", line)
            if kpt_match:
                # extract k-point index
                if band_index >= kpts_count:
                    raise ValueError(f"more k-points found than expected: {band_index} vs {kpts_count}")
                kpt = np.array([float(kpt_match.group(1)), float(kpt_match.group(2)), float(kpt_match.group(3))])
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
                    processed_energies = np.array([float(e) for e in re.findall(r"([-+]?\d*\.?\d+)", energy_line)])
                    all_point_energies[extracted_energies:extracted_energies + len(processed_energies)] = processed_energies
                    extracted_energies += len(processed_energies)
                    next_lc += 1
                if extracted_energies != kohn_sham_count:
                    raise ValueError(f"extracted {extracted_energies} band energies, expected {kohn_sham_count}")
                bands[band_index - 1] = all_point_energies

                # extract occupations
                next_lc += 1
                all_point_occupations = np.zeros(kohn_sham_count)
                extracted_occupations = 0
                while next_lc < len(lines):
                    occupation_line = lines[next_lc].strip()
                    if occupation_line == "":
                        break
                    processed_occupations = np.array([float(o) for o in re.findall(r"([-+]?\d*\.?\d+)", occupation_line)])
                    all_point_occupations[extracted_occupations:extracted_occupations + len(processed_occupations)] = processed_occupations
                    extracted_occupations += len(processed_occupations)
                    next_lc += 1
                if extracted_occupations != kohn_sham_count:
                    raise ValueError(f"extracted {extracted_occupations} occupations, expected {kohn_sham_count}")
                occupations[band_index - 1] = all_point_occupations
        lc += 1
    print(f"found {lc} lines in {os.path.relpath(filepath, start=PROJECT_DIR)} & extracted {band_index} k-points with {kohn_sham_count} bands each")

    return kpts, bands, occupations


if __name__ == "__main__":
    main()

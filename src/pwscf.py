import numpy as np
import json
import matplotlib.pyplot as plt


class PWSCFStruct:
    # intialization of sane defaults
    def __init__(self, file_path=None):
        self.BOHRtoA = 0.529177249
        self.RYtoeV = 13.6057039763
        self.program_version = ""
        self.volume = 0.0
        self.alat = 0.0
        self.nat = 0
        self.nelect = 0
        self.Ecut = 0.0
        self.RhoCut = 0.0
        self.Econv = 0.0
        self.beta = 0.0
        self.Exch = ""
        self.energy = 0.0
        self.natoms = 0
        self.bandgap = 0.0
        self.bands = 0
        self.lattice = {"a": np.zeros(3), "b": np.zeros(3), "c": np.zeros(3)}
        self.atoms = []
        self.norms = {"a": 0.0, "b": 0.0, "c": 0.0}
        self.angles = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        self.kpts = 0
        self.band_diagram = np.zeros(0)
        self.FermiTest = False
        self.Fermi = 0.0
        if file_path:
            self.process_file(file_path)

    # process the file and extract the data
    def process_file(self, file_path):
        f = open(file_path, "r")
        line_number = 0
        for i in f:
            if line_number < 1000:
                if "number of k points=" in i:
                    self.kpts = int(i.split()[4])
                    _ = next
                if "Program PWSCF" in i:
                    self.program_version = i.split()[2]
                    _ = next
                if "lattice parameter (alat)" in i:
                    self.alat = float(i.split()[4]) * self.BOHRtoA
                    _ = next
                if "number of Kohn-Sham states" in i:
                    self.bands = int(i.split()[4])
                if "unit-cell volume" in i and "new" not in i:
                    self.volume = float(i.split()[3]) * (self.BOHRtoA**3.0)
                    _ = next
                if "number of atoms/cell" in i:
                    self.natoms = int(i.split()[4])
                    _ = next
                if "number of atomic types" in i:
                    self.nat = int(i.split()[5])
                    _ = next
                if "number of electrons" in i:
                    self.nelect = float(i.split()[4])
                    _ = next
                if "kinetic-energy cutoff" in i:
                    self.Ecut = float(i.split()[3]) * self.RYtoeV
                    _ = next
                if "charge density cutoff" in i:
                    self.RhoCut = float(i.split()[4]) * self.RYtoeV
                    _ = next
                if "convergence threshold" in i:
                    self.Econv = float(i.split()[4])
                    _ = next
                if "mixing beta" in i:
                    self.beta = float(i.split()[3])
                    _ = next
                if "Exchange-correlation" in i:
                    self.Exch = i.split()[2]
                    _ = next
                if "a(1) =" in i:
                    tmp = i.split()
                    for j in range(0, 3):
                        self.lattice["a"][j] = float(tmp[j + 3])
                    _ = next
                if "a(2) =" in i:
                    tmp = i.split()
                    for j in range(0, 3):
                        self.lattice["b"][j] = float(tmp[j + 3])
                    _ = next
                if "a(3) =" in i:
                    tmp = i.split()
                    for j in range(0, 3):
                        self.lattice["c"][j] = float(tmp[j + 3])
                    _ = next
                if "site n.     atom                  positions (alat units)" in i:
                    for j in range(0, self.natoms):
                        line = next(f).split()
                        self.atoms.append(
                            [
                                line[1],
                                float(line[6]) * self.alat,
                                float(line[7]) * self.alat,
                                float(line[8]) * self.alat,
                            ]
                        )
                    _ = next
            if "!" in i:
                self.energy = float(i.split()[4]) * self.RYtoeV
            if "new unit-cell volume" in i:
                self.volume = float(i.split()[4]) * (self.BOHRtoA**3)
            if "Begin final coordinates" in i:
                # print(self.atoms)
                line = next(f)
                while "End final coordinates" not in line:
                    line = next(f)
                    if "CELL_PARAMETERS" in line:
                        self.alat = (
                            float(line.replace(")", "").split()[2]) * self.BOHRtoA
                        )
                        for j in ["a", "b", "c"]:
                            line = next(f)
                            tmp = line.split()
                            for k in range(0, 3):
                                self.lattice[j][k] = self.alat * float(tmp[k])
                    if "ATOMIC_POSITIONS" in line:
                        if "angstrom" in line:
                            for j in range(0, self.natoms):
                                line = next(f).split()
                                self.atoms[j] = [
                                    line[0],
                                    float(line[1]),
                                    float(line[2]),
                                    float(line[3]),
                                ]
            # parse the band data
            if (
                "End of self-consistent calculation" in i
            ):
                if np.floor(self.bands / 8.0) * 8.0 <= self.bands:
                    numlines = int(np.floor(self.bands / 8.0) + 1)
                    remainder = int(self.bands - np.floor(self.bands / 8.0) * 8.0)
                else:
                    numlines = int(np.floor(self.bands / 8.0))
                    remainder = 0
                self.band_diagram = np.zeros((self.kpts, self.bands))
                counter = 0
                while counter < self.kpts:
                    line = next(f)
                    if "k =" in line:
                        line = next(f)
                        counter1 = 0
                        for j in range(0, numlines):
                            line = next(f)
                            for k in range(0, len(line.split())):
                                self.band_diagram[counter][counter1 + k] = float(
                                    line.split()[k]
                                )
                            counter1 += 8
                        counter += 1
                _ = next
            if "highest occupied, lowest unoccupied level (ev)" in i:
                self.bandgap = float(i.split()[7]) - float(i.split()[6])
                _ = next
            # check for fermi energy
            if "the Fermi energy is" in i:
                self.Fermi = float(i.split()[4])
                self.FermiTest = True
                _ = next
            line_number += 1
        f.close()
        for vec in ["a", "b", "c"]:
            # cast the numpy scalar to a built-in float
            self.norms[vec] = float(np.linalg.norm(self.lattice[vec]))
            self.angles["alpha"] = (
                np.arccos(
                    np.dot(self.lattice["b"], self.lattice["c"])
                    / (self.norms["c"] * self.norms["b"])
                )
                * 180.0
                / np.pi
            )
        self.angles["gamma"] = (
            np.arccos(
                np.dot(self.lattice["a"], self.lattice["b"])
                / (self.norms["a"] * self.norms["b"])
            )
            * 180.0
            / np.pi
        )
        self.angles["beta"] = (
            np.arccos(
                np.dot(self.lattice["a"], self.lattice["c"])
                / (self.norms["a"] * self.norms["c"])
            )
            * 180.0
            / np.pi
        )
        if self.FermiTest:
            # center the band diagram around the fermi level
            self.band_diagram = np.subtract(self.band_diagram, self.Fermi)
            emin = np.zeros(self.kpts)
            emax = np.zeros(self.kpts)
            counter = 0
            for j in self.band_diagram:
                emin[counter] = j[np.searchsorted(j, 0.0, side="right") - 1]
                emax[counter] = j[np.searchsorted(j, 0.0, side="right")]
                counter += 1
            self.bandgap = float(np.min(emax - emin))

    def to_json(self):
        # Build a serializable dict without mutating internal arrays
        data = self.__dict__.copy()
        data["lattice"] = {k: v.tolist() for k, v in self.lattice.items()}
        if isinstance(self.band_diagram, np.ndarray):
            data["band_diagram"] = self.band_diagram.tolist()
        return json.dumps(
            data,
            default=lambda o: getattr(o, "__dict__", str(o)),
            sort_keys=True,
            indent=4,
        )

    def plot_bands(self):
        if self.band_diagram.size == 0:
            print("No band diagram data available.")
            return

        plt.figure(figsize=(10, 6))

        # plot fermi level
        plt.axhline(self.Fermi, color="black", linestyle="--", linewidth=0.5)

        for i in range(self.band_diagram.shape[1]):
            plt.plot(
                np.arange(self.kpts),
                self.band_diagram[:, i],
                label=f"Band {i + 1}",
                alpha=0.5,
        )

        plt.xticks(np.arange(self.kpts), np.arange(1, self.kpts + 1))
        plt.yticks(np.arange(-15, 10, 1))  # Adjust y-ticks as needed

        # plot data
        plt.title("Band Diagram")
        plt.xlabel("K-points")
        plt.ylabel("Energy (eV)")
        plt.legend(loc="upper right", fontsize="small")
        plt.grid()
        plt.tight_layout()
        plt.show()

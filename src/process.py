import numpy as np
import re
import os
import matplotlib.pyplot as plt

PROJECT_DIR = "./out/mp-149_si_p_doping"

def main():
    E_F = 0.0
    with open(os.path.join(PROJECT_DIR, "scf.out"), "r") as f:
        scf_text = f.read()
        m = re.search(r"the Fermi energy is\s+([+\-\d\.]+)\s+ev", scf_text)
        if m:
            E_F = float(m.group(1))
        else:
            print(f"[INFO] Fermi energy not found in {PROJECT_DIR}/scf.out")
            return
    print(f"[INFO] Fermi energy: {E_F} eV")



if __name__ == "__main__":
    main()

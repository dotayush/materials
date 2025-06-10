from pwscf import PWSCFStruct
import os

out_directory = os.path.join(os.getcwd(), "out", "mp-149_si_p_doping")
scf_output = PWSCFStruct(os.path.join(out_directory, "scf.out"))
print(scf_output.to_json())
scf_output.plot_bands()

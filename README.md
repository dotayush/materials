# materials.

"materials." is a collection of post-processing scripts to process and anaylze
materials data from Quantum ESPRESSO (QE) calculations.

### running.

#### setting up potentials.
1) download pseudopotentials from [SSSP precision library](https://www.materialscloud.org/discover/sssp/table/precision)
which contains all pseudopotential files and place them in some directory
(e.g. `./pseudos/`).
2) set the `PSEUDO_DIR` variable in `.env` file to the path of this directory.
3) run `python ./src/rename_pseudos.py` to rename the pseudopotential files
for other scripts to recognize.


#### creating pymatgen structure.
1) edit `./src/main.py`'s `main()` function to create/build/modify a `pymatgen`
structure object for your material of interest.
2) call `project_dir = setup_output_project("<project_name>")` to create a
project directory under `./out/` with the name `<project_name>`.
3) after your structure creation, call `generate_espresso_input(<my_structure>,
out_dir=project_dir, prefix="<espresso_output_prefix>")` to generate the input
filesfor Quantum ESPRESSO calculations under `./out/<project_name>/`.

the final code in main should look like this:
```python
from pymatgen.core import Structure

def main():
    # create your pymatgen structure object (below i've loaded a structure, but
    # you can create it from scratch or modify an existing one with pymatgen)
    my_structure = Structure.from_file("path/to/your/structure/file")

    # setup output project directory
    project_dir = setup_output_project("my_structure_project")

    # generate input files for Quantum ESPRESSO
    generate_espresso_input(my_structure, out_dir=project_dir, prefix="my_structure")
```

#### running calculations.
after generating the input files, you can run the calculations using
```bash

# run scf calculation
pw.x < ./out/<project_name>/scf.in > ./out/<project_name>/scf.out

# run nscf calculation
pw.x < ./out/<project_name>/nscf.in > ./out/<project_name>/nscf.out

# run dos calculation
pw.x < ./out/<project_name>/dos.in > ./out/<project_name>/dos.out

# run phonon calculation
ph.x < ./out/<project_name>/ph.in > ./out/<project_name>/ph.out
```

#### checking calculation progress.
run the following command in a separate terminal to monitor the output of
```bash
# calculation type = scf, nscf, dos or ph
tail -f ./out/<project_name>/<calculation_type>.out
```

### license.

       ╱|、
     (˚ˎ 。7
      |、˜〵
     じしˍ,)ノ

the repository and everything within is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
Refer to [COPYING.md](./COPYING.md) for the full license text.

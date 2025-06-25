# materials.

"materials." is a collection of post-processing scripts to process and anaylze
materials data from Quantum ESPRESSO (QE) calculations.

### running.

#### setting up potentials and environment variables.
1) download pseudopotentials from [SSSP precision library](https://www.materialscloud.org/discover/sssp/table/precision)
which contains all pseudopotential files and place them in some directory
(e.g. `./pseudos/`).
2) set the `PSEUDO_DIR` variable in `.env` file to the path of this directory.
3) run `python ./src/rename_pseudos.py` to rename the pseudopotential files
for other scripts to recognize.
4) copy `.env.example` to `.env` and set the `PROJECT_NAME`, leave the rest
to defaults.

#### generating input files.

you have to build your structure in `main.py`'s `main()` function.
you'll have to two more functions to generate the input files. here's
an example of how to do that:
```py
def main():
    # get project directory
    project_dir = setup_output_project(PROJECT_NAME)

    # build your structure here
    structure = get_structure_from_id("mp-149")

    # generate the input files
    generate_espresso_input(
        structure,
        out_dir=project_dir,
        prefix="si",
    )
```

#### running calculations.

after generating the input files, you can run the calculations using
```bash

# run scf calculation
pw.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/scf.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/scf.out

# run nscf calculation
pw.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/nscf.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/nscf.out

# run dos calculation
dos.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/dos.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/dos.out

# run phonon calculation
ph.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/ph.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/ph.out
```

#### checking calculation progress.

run the following command in a separate terminal to monitor the output of
```bash
# calculation type = scf, nscf, dos or ph
tail -f ./<OUTPUT_DIR>/<PROJECT_NAME>/<calculation_type>.out
```

#### switching exchange-correlation functional.

the files are generated with no exchange-correlation functional
specified. this means that the default functional of PBE is used
by Quantum ESPRESSO. to switch to another functional, you can
add these to `nscf.in` & `scf.in` files under the `&system` card
before running the calculations:
```bash
   input_dft        = 'hse'
   exx_fraction     = 0.25
   screening_parameter = 0.106
   ecutfock         = 120
```

`hse` should give you much accurate results at the cost of really long
calculation time. `pbe` is fast but not accurate. i don't recommend
simulating beyond 4 total atoms with `hse` functional and beyond
32 total atoms with `pbe` functional.

### license.

       ╱|、
     (˚ˎ 。7
      |、˜〵
     じしˍ,)ノ

the repository and everything within is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
Refer to [COPYING.md](./COPYING.md) for the full license text.

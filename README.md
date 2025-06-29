# materials.

"materials." is a small tool which generates the input files for Quantum Espresso.
it also contains a script to post-process the results of the calculations (scf,
nscf implemented so far) and extract useful information from them.

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
    structure = get_structure_from_id("mp-149") # Silicon ID from Materials Project

    # generate the input files
    generate_espresso_input(
        structure,
        out_dir=project_dir,
        prefix="si",
        like="insulator", # or "metal"; this decides what occupation scheme to use i.e. smearing (mv*) or fixed occupations;
        xc="pbe", # exchange-correlation functional; this sets the `input_dft` variable in the input files
        kpts=(7, 7, 7),
    )
```

then run,

```bash
python ./src/main.py
```


> although phonon files are generated for functionals other than `pbe`, they are not
> supported by Quantum ESPRESSO as of build 7.4.1 (which i used).

#### running calculations.

after generating the input files, you can run the calculations using
```bash

# run relaxation calculation
pw.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/relax.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/relax.out

# copy the relaxed structure to scf.in and nscf.in (bottom)

# run scf calculation
pw.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/scf.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/scf.out

# run nscf calculation
pw.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/nscf.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/nscf.out

# run dos calculation
dos.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/dos.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/dos.out

# run phonon calculation
ph.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/ph.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/ph.out
```

> you can check calculation progress by running `tail -f ./<OUTPUT_DIR>/<PROJECT_NAME>/<calculation_type>.out`
> in a separate terminal, where `<calculation_type>` is one of `scf`, `nscf`, `dos` or `ph`.

#### post-processing.

after running the calculations, you can post-process the results using
```bash
python ./src/process.py
```
this will generate `./<OUTPUT_DIR>/<PROJECT_NAME>/processed.json` file which contains
useful informaton extracted from scf and nscf calculations.

#### tips.

- you can use `mpirun -np <num_procs> pw.x < ./<OUTPUT_DIR>/<PROJECT_NAME>/scf.in > ./<OUTPUT_DIR>/<PROJECT_NAME>/scf.out`
to run the calculations in parallel on multiple processors.
- get your pseudopotentials from [here](https://pseudopotentials.quantum-espresso.org/).
- although phonon calculations are always generated, they are not supported by Quantum ESPRESSO as of
build 7.4.1 (which i used) for exchange-functionals other than `pbe`.
- certain exchange-functionals (eg. SCAN) require norm-conserving pseudopotentials to be used. i used
`sg15_oncv_upf_2020-02-06.tar.gz SG15 ONCV potentials in UPF format` from [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/)
for this purpose. you are still required to rename them and update your `.env` file if working with such
functionals.

### license.

       ╱|、
     (˚ˎ 。7
      |、˜〵
     じしˍ,)ノ

the repository and everything within is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
Refer to [COPYING.md](./COPYING.md) for the full license text.

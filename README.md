# material analysis.

the following code is used to build a doped strucutre using pymatgen and
then the structure files are converted from cif to pw.x input files for
running calculations in quantum espresso.

### running.

you'll need to edit `main()` to build your strucuture, change calculation
type and defining a project name. this will generate the input files for
scf, nscf and dos and phonon calculations.

```bash
# create quantum espresso input files
python ./src/main.py
```

i recommend downloading psudopotentials from [SSSP precision](https://www.materialscloud.org/discover/sssp/table/precision)
which contains all psuedopotential files and placing them in the `./pseudos/` directory.
i then recommend you to run,

```bash
# it removed the extra stuff and converts elements
# to lowercase.UPF filename. this matches the
# naming convention used in the input files for pseudos.
python ./src/rename_pseudos.py
```

now you're free to run your calculations in quantum espresso.

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

to check the status of the calculations,

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

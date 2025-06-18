"""
    "Materials" is a collection of post-processing scripts to process and
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
import os
from dotenv import load_dotenv
_ = load_dotenv()


PSEUDOS_DIR = os.path.join(os.getcwd(), os.getenv("PSEUDOS_DIR", "./psuedos"))

def rename_psuedo_files():
    cwd = os.getcwd()

    if not os.path.exists(PSEUDOS_DIR):
        print(f"[Error] directory {PSEUDOS_DIR} does not exist.")
        return

    updated_names = []
    for filename in os.listdir(PSEUDOS_DIR):
        # skip hidden files (those starting with a dot)
        if filename.startswith('.'):
            continue

        letters = filename[:3]
        parts = []
        if '.' in letters:
            parts = filename.split('.')
        elif '_' in letters:
            parts = filename.split('_')
        elif '-' in letters:
            parts = filename.split('-')
        else:
            print(f"Skipping '{filename}' as it does not contain '.' or '_' in the first three letters.")
            continue

        new_filename = "".join([parts[0].lower()]) + ".UPF"


        if new_filename in updated_names:
            print(f"Skipping '{filename}' as a file with the name '{new_filename}' already exists.")
            continue
        updated_names.append(new_filename)

        old_file_path = os.path.join(PSEUDOS_DIR, filename)
        new_file_path = os.path.join(PSEUDOS_DIR, new_filename)

        # os.rename(old_file_path, new_file_path)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")


if __name__ == "__main__":
    rename_psuedo_files()
    # print(os.getcwd())
    # print(os.listdir(os.getcwd()))
    # print(os.listdir(os.path.join(os.getcwd(), "psuedo")))

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

import os
from dotenv import load_dotenv

_ = load_dotenv()
MP_API_KEY = os.getenv("MP_API_KEY")
SAVED_STRUCTURES_DIR = os.path.join(
    os.getcwd(), os.getenv("SAVED_STRUCTURES_DIR", "./saved_structures")
)
PSEUDOPOTENTIALS_DIR = os.path.join(os.getcwd(), os.getenv("PSEUDOS_DIR", "./pseudos"))
OUTPUT_DIR = os.path.join(os.getcwd(), os.getenv("OUTPUT_DIR", "out"))
PROJECT_NAME = os.getenv("PROJECT_NAME", "default_project")
FUNCTIONAL = PROJECT_NAME.split("_")[1].upper()

if not FUNCTIONAL:
    raise ValueError("error: PROJECT_NAME must contain a functional name in the format 'project_functional'.")

def setup_output_project(project_name: str) -> str:
    """Setup the output directory for the project. If the directory already exists and has contents, it will error."""
    project_dir = os.path.join(os.getcwd(), OUTPUT_DIR, project_name)
    print(f"info: setting up project directory: {project_dir}")

    # error if the project directory already exists and has contents
    if os.path.exists(project_dir) and len(os.listdir(project_dir)) > 0:
        raise FileExistsError(
            f"error: project directory {project_dir} already exists and has contents. please remove the directory or choose a different project name."
        )

    os.makedirs(project_dir, exist_ok=True)
    return project_dir

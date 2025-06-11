import os

# the goal is to iterate through cwd + psuedo folder for all psuedo files
# then in the first three letters of the filename, either check for . or _ and then
# split the filename into whatever is before the first . or _ and then
# change the first part to lowercase

PSEUDOS_DIR = os.path.join(os.getcwd(), "pseudos")

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

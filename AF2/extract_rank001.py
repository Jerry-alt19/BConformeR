import os
import shutil

src_folder = '...' # directory containing multiple prediction directories
dst_folder = '...'

os.makedirs(dst_folder, exist_ok=True)

count_copied = 0
count_deleted = 0

# Traverse all subdirectories and files
for root, dirs, files in os.walk(src_folder):
    for file in files:
        full_path = os.path.join(root, file)

        is_pdb = file.lower().endswith('.pdb')
        contains_rank001 = 'rank_001' in file

        if is_pdb and contains_rank001:
            # Copy the file to the destination folder (flattened structure)
            dst_path = os.path.join(dst_folder, file)
            shutil.copy2(full_path, dst_path)
            count_copied += 1
        else:
            # Delete non-target files
            try:
                os.remove(full_path)
                count_deleted += 1
            except Exception as e:
                print(f"Failed to delete {full_path}: {e}")

print(f"\nSuccessfully copied {count_copied} '.pdb' files containing 'rank_001' to '{dst_folder}' (flat structure).")
print(f"Successfully deleted {count_deleted} non-target files (non-.pdb or not containing 'rank_001').")

import os
import shutil

# --- Configuration ---
BASE_FOLDER = "demos"
# The specific subfolders you want to organize
# TOURNAMENTS = ["Antwerp", "Rio", "Stockholm"]
TOURNAMENTS = ["Antwerp"]

# --- Main Script ---
print(f"Starting organization for: {', '.join(TOURNAMENTS)}...\n")

# Get the absolute path of the base 'demos' folder
base_path = os.path.abspath(BASE_FOLDER)

for tournament in TOURNAMENTS:
    # Build the full path to the specific tournament folder (e.g., .../demos/Antwerp)
    tournament_path = os.path.join(base_path, tournament)

    # Check if the folder actually exists before trying to organize it
    if not os.path.exists(tournament_path):
        print(f"⚠️  Skipping '{tournament}' (Folder not found)")
        continue

    print(f"📂 Processing '{tournament}'...")

    # count moved files for this folder
    moved_count = 0

    # Walk through the tournament folder
    for dirpath, dirnames, filenames in os.walk(tournament_path):

        # Skip the root tournament folder itself (we don't need to move files that are already there)
        if dirpath==tournament_path:
            continue

        for filename in filenames:
            # Check for .dem files (and .rar if you haven't extracted them yet)
            if filename.endswith(".dem") or filename.endswith(".rar"):

                # Source: where the file is now (deep inside a subfolder)
                source_path = os.path.join(dirpath, filename)

                # Destination: the main tournament folder
                destination_path = os.path.join(tournament_path, filename)

                # Check if a file with the same name already exists in the destination
                if not os.path.exists(destination_path):
                    shutil.move(source_path, destination_path)
                    print(f"   -> Moved: {filename}")
                    moved_count += 1
                else:
                    print(f"   -> Skipped (Duplicate): {filename}")

    if moved_count==0:
        print("   (No files needed moving)")
    else:
        print(f"   Done! Moved {moved_count} files.")
    print("-" * 30)

print("\nAll requested folders have been organized!")
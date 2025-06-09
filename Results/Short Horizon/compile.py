import os
import csv
import torch

# Get the directory where compile.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# List of target subfolders
target_folders = ["timegpt"]

# Store all file paths
all_file_paths = []

for folder in target_folders:
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                all_file_paths.append(full_path)

# Print or use the file paths
# for path in all_file_paths:
#     print(path)

map_dict = {}
for path in all_file_paths:
    if "preds" in path:
        all_rows = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                all_rows.append(row)
        for i,row in enumerate(all_rows):
            map_dict[os.path.basename(path).split(" ")[0] + " " + os.path.basename(row[0])[:-5] + " " + 'preds'] = row[1:]
    elif "actuals" in path:
        all_rows = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                all_rows.append(row)
        for i,row in enumerate(all_rows):
            map_dict[os.path.basename(path).split(" ")[0] + " " + os.path.basename(row[0])[:-5] + " " + 'actuals'] = row[1:]

torch.save(map_dict, "short_timegpt_preds_actuals.pt")
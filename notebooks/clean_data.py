# remove faulty trajectories near cone

import pandas as pd
import os

# Load the trajectory data
data_path = "../data/trajectories/trajectories_200k_v2.txt"
data = pd.read_csv(data_path, header=None, delim_whitespace=True)
data.columns = ['X', 'Y', 'yaw', 'speed', 'bool_end']

# Filter and mark runs to be removed
rows_to_remove = []
current_run = []
x_wall, y_wall, proximity_threshold = 26.775, 19.3, 0.25

for index, row in data.iterrows():
    current_run.append(index)
    if row['bool_end'] == 1:
        for idx in current_run:
            if ((abs(data.loc[idx, 'X'] - x_wall) < proximity_threshold) or 
                (abs(data.loc[idx, 'Y'] - y_wall) < proximity_threshold)):
                rows_to_remove.extend(current_run)
                break
        current_run = []

# Create the cleaned data file
cleaned_data = data.drop(rows_to_remove)
cleaned_data.to_csv('cleaned_trajectories_200k.txt', sep=' ', header=False, index=False)

# remove images

for idx in rows_to_remove:
    image_path = f"image_data/{str(idx+1).zfill(4)}.jpg"
    if os.path.exists(image_path):
        os.remove(image_path)

# Rename Remaining Images

existing_images = sorted([f for f in os.listdir('image_data') if f.endswith('.jpg')])
for idx, image_name in enumerate(existing_images):
    old_path = os.path.join('image_data', image_name)
    new_path = os.path.join('image_data', f"{str(idx+1).zfill(4)}.jpg")
    os.rename(old_path, new_path)

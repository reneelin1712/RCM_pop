import pandas as pd

# Load the CSV file
file_path = "trajectory_with_timestep.csv"
df = pd.read_csv(file_path)

# Task 1: Add a "diff" column
df['diff'] = df.apply(lambda row: '100%' if row['Test Trajectory'] == row['Learner Trajectory'] else '', axis=1)

# Extract origin and destination for each trajectory
def get_origin_destination(trajectory):
    links = trajectory.split('_')
    return links[0], links[-1]

df['Origin'], df['Destination'] = zip(*df['Test Trajectory'].apply(get_origin_destination))

# Task 2: Find same OD with different paths
# Group by 'Origin' and 'Destination', and keep only groups with more than one unique path
od_groups = df.groupby(['Origin', 'Destination']).filter(lambda x: x['Test Trajectory'].nunique() > 1)

# Save the filtered groups with same OD and different paths to an Excel file
output_file = "same_OD_different_paths.xlsx"
od_groups.to_excel(output_file, index=False, engine='openpyxl')

print(f"Process complete. Results saved to {output_file}")

import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class WWADL_Dataset:
    def __init__(self, root_path='/data/WWADL/processed_data'):
        self.root_path = root_path
        self.imu_path = os.path.join(root_path, 'imu')
        self.wifi_path = os.path.join(root_path, 'wifi')
        self.airpods_path = os.path.join(root_path, 'AirPodsPro')

    def generate_file_list(self, volunteer_ids=range(16), scene_ids=range(1, 4), action_ids=range(1, 21)):
        file_records = []

        # Iterate through volunteer IDs, scene IDs, and action IDs
        for volunteer_id in volunteer_ids:  # Default: 0-15
            for scene_id in scene_ids:  # Default: 1-3
                for action_id in action_ids:  # Default: 1-20
                    # Generate the file name
                    file_name = f"{volunteer_id}_{scene_id}_{action_id}.h5"

                    # Check file existence in each folder
                    imu_file = os.path.join(self.imu_path, file_name)
                    wifi_file = os.path.join(self.wifi_path, file_name)
                    airpods_file = os.path.join(self.airpods_path, file_name)

                    if os.path.exists(imu_file) and os.path.exists(wifi_file) and os.path.exists(airpods_file):
                        # Append record if all files are found
                        file_records.append({
                            'volunteer_id': volunteer_id,
                            'scene_id': scene_id,
                            'action_group_id': action_id,
                            'imu_path': imu_file,
                            'wifi_path': wifi_file,
                            'airpods_path': airpods_file
                        })

        return file_records

    def find_missing_airpods_files(self):
        missing_files = []

        for volunteer_id in range(16):
            for scene_id in range(1, 4):
                for action_id in range(1, 21):
                    file_name = f"{volunteer_id}_{scene_id}_{action_id}.h5"

                    imu_file = os.path.join(self.imu_path, file_name)
                    wifi_file = os.path.join(self.wifi_path, file_name)
                    airpods_file = os.path.join(self.airpods_path, file_name)

                    if os.path.exists(imu_file) and os.path.exists(wifi_file) and not os.path.exists(airpods_file):
                        missing_files.append({
                            'volunteer_id': volunteer_id,
                            'scene_id': scene_id,
                            'action_group_id': action_id,
                            'imu_path': imu_file,
                            'wifi_path': wifi_file
                        })

        return missing_files

    def count_by_volunteer_and_scene(self):
        file_records = self.generate_file_list()
        count_dict = defaultdict(lambda: defaultdict(int))

        for record in file_records:
            volunteer_id = record['volunteer_id']
            scene_id = record['scene_id']
            count_dict[volunteer_id][scene_id] += 1

        return count_dict

    def plot_counts_by_scene(self):
        counts = self.count_by_volunteer_and_scene()
        scene_counts = defaultdict(int)

        # Aggregate counts by scene
        for volunteer_id, scenes in counts.items():
            for scene_id, count in scenes.items():
                scene_counts[scene_id] += count

        # Sort scenes for consistent plotting
        sorted_scenes = sorted(scene_counts.keys())
        sorted_counts = [scene_counts[scene] for scene in sorted_scenes]

        # Plot
        plt.bar(sorted_scenes, sorted_counts)
        plt.xlabel('Scene ID')
        plt.ylabel('File Count')
        plt.title('File Counts by Scene')
        plt.xticks(sorted_scenes)
        plt.show()

    def plot_heatmap_by_volunteer_and_scene(self):
        counts = self.count_by_volunteer_and_scene()
        heatmap_data = np.zeros((16, 3))  # 16 volunteers, 3 scenes

        # Populate the heatmap data
        for volunteer_id, scenes in counts.items():
            for scene_id, count in scenes.items():
                heatmap_data[volunteer_id, scene_id - 1] = count

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", xticklabels=["Scene 1", "Scene 2", "Scene 3"], yticklabels=[f"Volunteer {i}" for i in range(16)])
        plt.xlabel("Scene")
        plt.ylabel("Volunteer")
        plt.title("File Counts by Volunteer and Scene")

        # Save the heatmap to a file
        heatmap_file_path = os.path.join(self.root_path, 'heatmap_by_volunteer_and_scene.png')
        plt.savefig(heatmap_file_path)
        print(f"Heatmap saved to {heatmap_file_path}")
        plt.close()

    def write_to_csv(self, output_csv='output.csv'):
        # Generate file list
        file_records = self.generate_file_list()

        # Define CSV headers
        headers = ['volunteer_id', 'scene_id', 'action_group_id', 'imu_path', 'wifi_path', 'airpods_path']

        # Write to CSV
        output_csv = os.path.join(self.root_path, output_csv)
        with open(output_csv, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(file_records)

        print(f"CSV file written to {output_csv}.")

# Example usage
if __name__ == "__main__":
    dataset = WWADL_Dataset(root_path='/data/WWADL/processed_data')
    dataset.write_to_csv(output_csv='output.csv')

    # Count and print statistics
    counts = dataset.count_by_volunteer_and_scene()
    for volunteer_id, scenes in counts.items():
        for scene_id, count in scenes.items():
            print(f"Volunteer {volunteer_id}, Scene {scene_id}: {count} files")

    # Plot counts by scene
    dataset.plot_counts_by_scene()

    # Plot heatmap of counts by volunteer and scene
    dataset.plot_heatmap_by_volunteer_and_scene()

    # Find and print missing AirPods files
    missing_files = dataset.find_missing_airpods_files()
    print("Files missing in AirPods:")
    for file in missing_files:
        print(file)

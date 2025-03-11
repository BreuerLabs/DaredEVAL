# import matplotlib.pyplot as plt
# import numpy as np
# import wandb
# import csv
# import argparse

# def read_saved_metrics(filename):
#     filename = f"{filename}"
#     with open(filename, 'r', newline='') as csv_file:
#         reader = csv.reader(csv_file)
#         precision_list = [row for row in reader] 
#         precision_list = [(float(target), float(distance)) for target, distance in precision_list[1:]] # Convert to floats if necessary
#     return precision_list

# # Argument parser for command-line arguments
# parser = argparse.ArgumentParser(description='Plot PNP Histograms')
# parser.add_argument('run_id', type=str, help='The run ID for the wandb run')
# args = parser.parse_args()
# run_id = args.run_id

# # Initialize a run using the run ID
# bins = 20
# api = wandb.Api()
# run = api.run('plug_and_play/' + run_id)

# # Download a specific file from the run
# mean_distances_list_inception_path = f'model_inversion/plug_and_play/results/distance_inceptionv3_list_filtered/{run_id}.csv'
# mean_distances_list_facenet_path = f'model_inversion/plug_and_play/results/distance_facenet_list_filtered/{run_id}.csv'

# run.file(mean_distances_list_inception_path ).download(replace=True).name
# run.file(mean_distances_list_facenet_path ).download(replace=True).name

# mean_distances_list_inception = read_saved_metrics(mean_distances_list_inception_path)
# mean_distances_list_facenet = read_saved_metrics(mean_distances_list_facenet_path)

# print(f"Downloaded file to {mean_distances_list_inception_path}")
# print(f"Downloaded file to {mean_distances_list_facenet_path}")

# # Plot the m 
# plt.clf()
# distances_arr_inception = np.array([mean_distances_list_inception[i][1] for i in range(len(mean_distances_list_inception))])

# plt.figure(figsize=(10, 6))
# plt.hist(distances_arr_inception, bins=bins, edgecolor='black')
# plt.xlabel('Feature Distance')
# plt.ylabel('Frequency')
# plt.title('Distribution of Feauture Distances (Inception)')

# plt.savefig(f'feature_hist_inception_{run_id}.pdf', format = "pdf")

# plt.clf()
# distances_arr_facenet = np.array([mean_distances_list_facenet[i][1] for i in range(len(mean_distances_list_facenet))])

# plt.figure(figsize=(10, 6))
# plt.hist(distances_arr_facenet, bins=bins, edgecolor='black')
# plt.xlabel('Feature Distance')
# plt.ylabel('Frequency')
# plt.title('Distribution of Feauture Distances (FaceNet)')

# plt.savefig(f'feature_hist_facenet_{run_id}.pdf', format = "pdf")
    
import matplotlib.pyplot as plt
import numpy as np
import wandb
import csv
import argparse

def read_saved_metrics(filename):
    with open(filename, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        precision_list = [row for row in reader] 
        precision_list = [(float(target), float(distance)) for target, distance in precision_list[1:]] # Convert to floats if necessary
    return precision_list

def plot_histogram(data, bins, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', color=color, alpha=0.8)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    
    return plt

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Plot PNP Histograms')
parser.add_argument('run_id', type=str, help='The run ID for the wandb run')
args = parser.parse_args()
run_id = args.run_id

# Initialize a run using the run ID
bins = 20
api = wandb.Api()
run = api.run('plug_and_play/' + run_id)

# Download specific files from the run
mean_distances_list_inception_path = f'model_inversion/plug_and_play/results/distance_inceptionv3_list_filtered/{run_id}.csv'
mean_distances_list_facenet_path = f'model_inversion/plug_and_play/results/distance_facenet_list_filtered/{run_id}.csv'

run.file(mean_distances_list_inception_path).download(replace=True).name
run.file(mean_distances_list_facenet_path).download(replace=True).name

mean_distances_list_inception = read_saved_metrics(mean_distances_list_inception_path)
mean_distances_list_facenet = read_saved_metrics(mean_distances_list_facenet_path)

print(f"Downloaded file to {mean_distances_list_inception_path}")
print(f"Downloaded file to {mean_distances_list_facenet_path}")

# Prepare data for plotting
distances_arr_inception = np.array([mean_distances_list_inception[i][1] for i in range(len(mean_distances_list_inception))])
distances_arr_facenet = np.array([mean_distances_list_facenet[i][1] for i in range(len(mean_distances_list_facenet))])

# Plot histograms
plt = plot_histogram(distances_arr_inception, bins, 'Distribution of Feature Distances (Inception)', 
               'Feature Distance', 'Frequency', color = 'lightcoral')

plt.savefig(f'feature_hist_inception_{run_id}.pdf', format="pdf")
plt.clf()

plt = plot_histogram(distances_arr_facenet, bins, 'Distribution of Feature Distances (FaceNet)', 
               'Feature Distance', 'Frequency', 'skyblue')

plt.savefig(f'feature_hist_facenet_{run_id}.pdf', format="pdf")
plt.clf()
import matplotlib
matplotlib.use("Agg")  # GUI 없이 이미지를 저장하도록 설정

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_distances_grassmannian(motion_data):
    """
    Calculate Euclidean and Grassmannian Geodesic distances for the given motion data.
    """
    euclidean_dists = []
    grassmannian_dists = []

    for i in range(len(motion_data) - 1):
        # Calculate Euclidean distance
        euclidean_dist = torch.norm(motion_data[i + 1] - motion_data[i], p=2).item()
        euclidean_dists.append(euclidean_dist)

        # Compute Grassmannian geodesic distance
        X1 = motion_data[i]
        X2 = motion_data[i + 1]

        # Compute covariance matrices
        C1 = torch.matmul(X1.T, X1)
        C2 = torch.matmul(X2.T, X2)

        # Perform eigen-decomposition to form subspaces
        U1, _, _ = torch.svd(C1)
        U2, _, _ = torch.svd(C2)

        # Compute principal angles using SVD
        svd_values = torch.linalg.svdvals(torch.matmul(U1.T, U2))
        svd_values = torch.clamp(svd_values, -1.0, 1.0)  # Clamp to avoid numerical errors
        principal_angles = torch.acos(svd_values)

        # Geodesic distance is the L2 norm of principal angles
        grassmannian_dist = torch.norm(principal_angles).item()
        grassmannian_dists.append(grassmannian_dist)

    return euclidean_dists, grassmannian_dists


def visualize_motion(motion_data):
    """
    Visualize the motion data in 3D space and save it as an image.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        motion_data[:, 0].numpy(),
        motion_data[:, 1].numpy(),
        zs=range(len(motion_data)),
        marker='o'
    )
    ax.set_title("Robot Motion in 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time Step")

    output_path = "robot_motion_3d.png"
    plt.savefig(output_path)
    print(f"3D motion visualization saved as '{output_path}'.")
    plt.close(fig)

def load_data():
    """
    Load the motion data from the specified file.
    """
    path = "bet_data_release/blockpush"  # Update this path if necessary
    motion_data = np.load(f"{path}/multimodal_push_actions.npy")

    # Convert to torch tensors for consistency
    motion_data = torch.tensor(motion_data, dtype=torch.float32)

    return motion_data
def calculate_distances_with_projection(motion_data):
    """
    Calculate Euclidean, Projection, and Grassmannian Geodesic distances for the given motion data.
    """
    euclidean_dists = []
    projection_dists = []
    grassmannian_dists = []

    for i in range(len(motion_data) - 1):
        # Calculate Euclidean distance
        euclidean_dist = torch.norm(motion_data[i + 1] - motion_data[i], p=2).item()
        euclidean_dists.append(euclidean_dist)

        # Ensure motion_data[i] and motion_data[i+1] have correct shape
        X1 = motion_data[i].unsqueeze(0) if motion_data[i].dim() < 2 else motion_data[i]
        X2 = motion_data[i + 1].unsqueeze(0) if motion_data[i + 1].dim() < 2 else motion_data[i + 1]

        # Compute projection matrices
        P1 = torch.matmul(X1.T, X1) / torch.norm(X1)
        P2 = torch.matmul(X2.T, X2) / torch.norm(X2)

        # Frobenius norm of projection difference
        projection_dist = torch.norm(P1 - P2, p='fro').item()
        projection_dists.append(projection_dist)

        # Compute Grassmannian geodesic distance
        U1, _, _ = torch.svd(P1)
        U2, _, _ = torch.svd(P2)

        # Principal angles via SVD
        svd_values = torch.linalg.svdvals(torch.matmul(U1.T, U2))
        svd_values = torch.clamp(svd_values, -1.0, 1.0)  # Clamp to avoid numerical errors
        principal_angles = torch.acos(svd_values)

        # Geodesic distance is the L2 norm of principal angles
        grassmannian_dist = torch.norm(principal_angles).item()
        grassmannian_dists.append(grassmannian_dist)

    return euclidean_dists, projection_dists, grassmannian_dists

def visualize_combined_distributions(euclidean_dists, projection_dists, grassmannian_dists):
    """
    Visualize distributions of Euclidean, Projection, and Grassmannian distances in separate histograms.
    """
    # Euclidean distances
    fig = plt.figure(figsize=(8, 6))
    plt.hist(euclidean_dists, bins=20, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Euclidean Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("euclidean_distance_distribution.png")
    print("Euclidean distance distribution saved as 'euclidean_distance_distribution.png'.")
    plt.close(fig)

    # Projection distances
    fig = plt.figure(figsize=(8, 6))
    plt.hist(projection_dists, bins=20, alpha=0.7, color="purple", edgecolor="black")
    plt.title("Projection Metric Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("projection_distance_distribution.png")
    print("Projection metric distance distribution saved as 'projection_distance_distribution.png'.")
    plt.close(fig)

    # Grassmannian distances
    fig = plt.figure(figsize=(8, 6))
    plt.hist(grassmannian_dists, bins=20, alpha=0.7, color="green", edgecolor="black")
    plt.title("Grassmannian Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("grassmannian_distance_distribution.png")
    print("Grassmannian distance distribution saved as 'grassmannian_distance_distribution.png'.")
    plt.close(fig)

def visualize_combined_distributions_subplot(euclidean_dists, projection_dists, grassmannian_dists, output_path="clustering_comparison.png"):
    """
    Visualize distributions of Euclidean, Projection, and Grassmannian distances in a single subplot figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Euclidean distances
    axs[0].hist(euclidean_dists, bins=20, alpha=0.7, color="blue", edgecolor="black")
    axs[0].set_title("Euclidean Distance Distribution")
    axs[0].set_xlabel("Distance")
    axs[0].set_ylabel("Frequency")

    # Projection distances
    axs[1].hist(projection_dists, bins=20, alpha=0.7, color="purple", edgecolor="black")
    axs[1].set_title("Projection Metric Distance Distribution")
    axs[1].set_xlabel("Distance")
    axs[1].set_ylabel("Frequency")

    # Grassmannian distances
    axs[2].hist(grassmannian_dists, bins=20, alpha=0.7, color="green", edgecolor="black")
    axs[2].set_title("Grassmannian Distance Distribution")
    axs[2].set_xlabel("Distance")
    axs[2].set_ylabel("Frequency")

    # Save the combined figure
    plt.suptitle("Comparison of Distance Metrics")
    plt.savefig(output_path)
    print(f"Clustering comparison figure saved as '{output_path}'.")
    plt.close(fig)


if __name__ == "__main__":
    try:
        # Load motion data
        motion_data = load_data()

        # Calculate distances using Euclidean, Projection, and Grassmannian metrics
        euclidean_dists, projection_dists, grassmannian_dists = calculate_distances_with_projection(motion_data[:, 0, :])

        # Visualize motion and combined distributions
        visualize_motion(motion_data[:, 0, :])
        visualize_combined_distributions_subplot(euclidean_dists, projection_dists, grassmannian_dists)
    except Exception as e:
        print(f"An error occurred: {e}")
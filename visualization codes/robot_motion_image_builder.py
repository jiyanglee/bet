import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# 데이터 불러오기
path = "bet_data_release/blockpush"
motion_data = torch.tensor(np.load(os.path.join(path, "multimodal_push_actions.npy")), dtype=torch.float32)

# 이미지 저장 폴더 생성
output_folder = "motion_data_images"
os.makedirs(output_folder, exist_ok=True)

def calculate_geodesic_distance(p1, p2):
    """
    2차원 벡터 사이의 Geodesic 거리를 계산합니다.
    """
    cos_theta = torch.clamp(torch.dot(p1, p2) / (torch.norm(p1) * torch.norm(p2)), -1.0, 1.0)
    geodesic_distance = torch.acos(cos_theta).item()
    return geodesic_distance

def calculate_distances(sequence):
    """
    주어진 모션 시퀀스에 대해 Euclidean 및 Geodesic 거리를 계산합니다.
    """
    euclidean_dists = []
    geodesic_dists = []

    for j in range(len(sequence) - 1):
        # Euclidean 거리 계산
        euclidean_dist = torch.norm(sequence[j + 1] - sequence[j]).item()
        euclidean_dists.append(euclidean_dist)

        # Geodesic 거리 계산
        geodesic_dist = calculate_geodesic_distance(sequence[j], sequence[j + 1])
        geodesic_dists.append(geodesic_dist)

    return euclidean_dists, geodesic_dists

def save_motion_images_with_distances(motion_data, output_folder):
    """
    각 모션 시퀀스를 시각화하여 Geodesic 및 Euclidean 거리를 포함한 이미지를 생성하고 저장합니다.
    """
    num_sequences = motion_data.shape[0]

    for i in range(num_sequences):
        sequence = motion_data[i]
        euclidean_dists, geodesic_dists = calculate_distances(sequence)

        # 2D 모션 데이터 시각화
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # 2D 모션 데이터 플롯
        ax[0].plot(sequence[:, 0], sequence[:, 1], marker='o', label="Motion")
        ax[0].set_title(f"Motion {i} (2D Trajectory)")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")
        ax[0].legend()

        # Geodesic 및 Euclidean 거리 플롯
        ax[1].plot(range(len(geodesic_dists)), geodesic_dists, label="Geodesic Distance", color="blue")
        ax[1].plot(range(len(euclidean_dists)), euclidean_dists, label="Euclidean Distance", color="green")
        ax[1].set_title(f"Motion {i} (Distances)")
        ax[1].set_xlabel("Time Step")
        ax[1].set_ylabel("Distance")
        ax[1].legend()

        # 이미지 저장
        output_path = os.path.join(output_folder, f"motion_{i}.png")
        plt.savefig(output_path)
        plt.close(fig)

# 실행
save_motion_images_with_distances(motion_data, output_folder)

print(f"Images saved to folder: {output_folder}")
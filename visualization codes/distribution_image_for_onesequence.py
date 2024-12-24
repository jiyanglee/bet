import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# 데이터 로드
path = "bet_data_release/blockpush"
actions = np.load(os.path.join(path, "multimodal_push_actions.npy"))
masks = np.load(os.path.join(path, "multimodal_push_masks.npy"))
observations = np.load(os.path.join(path, "multimodal_push_observations.npy"))

# 데이터를 torch 텐서로 변환
observations = torch.tensor(observations, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
masks = torch.tensor(masks, dtype=torch.bool)

# 데이터 구조 확인
n_data, sequence_length, obs_dim = observations.shape
k = 10  # Grassmann 다양체의 차원

# 1. Grassmann 다양체 상의 매핑
X = observations  # 관측 데이터를 사용, (n_data, sequence_length, obs_dim)
X_flattened = X.view(n_data, sequence_length, -1)

# 공분산 행렬 계산
covariance_matrix = torch.matmul(X_flattened.transpose(1, 2), X_flattened)
Lambda, Y = torch.linalg.eig(covariance_matrix)

# 실제 값만 사용
Lambda_real = Lambda.real
Y_real = Y.real

# 상위 k개의 고유값 및 고유벡터 선택
sorted_indices = Lambda_real.argsort(dim=1, descending=True)
top_k_indices = sorted_indices[:, :k]
Lambda_top = Lambda_real.gather(1, top_k_indices)
Y_top = torch.gather(Y_real, 2, top_k_indices.unsqueeze(1).expand(-1, Y_real.size(1), -1))

# 투영 행렬 계산
P = torch.matmul(Y_top, Y_top.transpose(1, 2))

# 2. 거리 계산 (지오데식 거리 및 유클리드 거리)
def geodesic_distance(P1, P2):
    _, singular_values, _ = torch.linalg.svd(torch.matmul(P1.transpose(-2, -1), P2))
    principal_angles = torch.acos(torch.clamp(singular_values, -1.0, 1.0))
    return torch.norm(principal_angles, dim=-1)

def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, p=2)

# 3. 특정 시퀀스의 거리 분포
sequence_index = 40
def analyze_sequence_distances(sequence_index, X_flattened, P, sequence_length):
    geodesic_dists = []
    euclidean_dists = []

    for j in range(sequence_length - 1):
        geodesic_dists.append(geodesic_distance(P[sequence_index], P[sequence_index]).item())
        euclidean_dists.append(euclidean_distance(X_flattened[sequence_index, j], X_flattened[sequence_index, j + 1]).item())

    # Geodesic Distance Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(geodesic_dists, bins=20, alpha=0.7, label="Geodesic Distance", color='blue')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title(f"Geodesic Distance Distribution for Sequence {sequence_index}")
    plt.legend()
    plt.savefig(f"geodesic_distance_distribution_sequence_{sequence_index}.png")

    # Euclidean Distance Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(euclidean_dists, bins=20, alpha=0.7, label="Euclidean Distance", color='green')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title(f"Euclidean Distance Distribution for Sequence {sequence_index}")
    plt.legend()
    plt.savefig(f"euclidean_distance_distribution_sequence_{sequence_index}.png")

analyze_sequence_distances(sequence_index, X_flattened, P, sequence_length)

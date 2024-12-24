import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

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

# 2. 거리 계산 (지오데식 거리 및 프로젝션 거리)
def geodesic_distance(P1, P2):
    _, singular_values, _ = torch.linalg.svd(torch.matmul(P1.transpose(-2, -1), P2))
    principal_angles = torch.acos(torch.clamp(singular_values, -1.0, 1.0))
    return torch.norm(principal_angles, dim=-1)

def projection_distance(P1, P2):
    return torch.norm(P1 - P2, p='fro')

def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, p=2)

# 순서 기반 거리 계산
seq_distance_proj = []
seq_distance_geo = []
seq_distance_euc = []

for i in range(n_data):
    for j in range(sequence_length - 1):
        seq_distance_proj.append(projection_distance(P[i], P[i]).item())
        seq_distance_geo.append(geodesic_distance(P[i], P[i]).item())
        seq_distance_euc.append(euclidean_distance(X_flattened[i, j], X_flattened[i, j+1]).item())

# 3. 거리 분포 시각화
# KDE 시각화를 추가하여 분포의 해상도를 높임
def visualize_distance_distributions(distance_list, title):
    plt.figure(figsize=(8, 6))
    plt.hist(distance_list, bins=50, color='blue', alpha=0.6, label=title)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {title}")
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(f"Distribution of {title}")
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")

# 거리 분포 시각화
visualize_distance_distributions(seq_distance_proj, "Sequential Projection Distance")
visualize_distance_distributions(seq_distance_geo, "Sequential Geodesic Distance")
visualize_distance_distributions(seq_distance_euc, "Sequential Euclidean Distance")

# 4. 시각화: 투영 행렬의 주요 성분
# Sequence를 PCA 로 시각화
# 시각적 구분성을 높이기 위해 각각 다른 색상과 투명도를 추가

def visualize_projections_sequence(Y_top):
    from sklearn.decomposition import PCA

    plt.figure(figsize=(10, 8))
    for i in range(5):  # 첫 5개의 시퀀스만 시각화
        Y_flattened = Y_top[i].detach().numpy()
        pca = PCA(n_components=2)
        Y_pca = pca.fit_transform(Y_flattened)

        plt.plot(Y_pca[:, 0], Y_pca[:, 1], marker='o', label=f"Sequence {i}", alpha=0.7)

    plt.title("Projection Visualization per Sequence")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig("projection_sequence_visualization.png")

visualize_projections_sequence(Y_top)

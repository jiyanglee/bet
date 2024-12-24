import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models.action_ae.discretizers.k_means import KMeansDiscretizer
from models.action_ae.discretizers.manifold import ManifoldDiscretizer

BIN_NUM=96

# 데이터 로드
path = "/root/bet/bet_data_release/blockpush/multimodal_push_actions.npy"
actions = torch.tensor(np.load(path), dtype=torch.float32)
print(f"Loaded actions shape: {actions.shape}")

# 클러스터링 클래스 초기화
action_dim = actions.shape[-1]
kmeans_discretizer = KMeansDiscretizer(action_dim=action_dim, num_bins=BIN_NUM, device="cpu")
manifold_discretizer = ManifoldDiscretizer(action_dim=action_dim, num_bins=BIN_NUM, device="cuda")

# 클러스터링 수행
print("Fitting KMeansDiscretizer...")
kmeans_discretizer.fit_discretizer(actions)
print("Fitting ManifoldDiscretizer...")
manifold_discretizer.fit_discretizer(actions)

# TSNE 시각화를 위한 데이터 준비
def prepare_tsne_data(actions, discretizer, name):
    flattened_actions = actions.view(-1, actions.shape[-1])
    labels = discretizer.encode_into_latent(flattened_actions).view(-1).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, verbose=1)
    tsne_result = tsne.fit_transform(flattened_actions.cpu().numpy())
    return tsne_result, labels, name

# KMeans 결과 준비
kmeans_tsne, kmeans_labels, kmeans_name = prepare_tsne_data(actions, kmeans_discretizer, "KMeans")

# Manifold 결과 준비
manifold_tsne, manifold_labels, manifold_name = prepare_tsne_data(actions, manifold_discretizer, "Grassmannian")

# 시각화 함수
def plot_clustering(tsne_result, labels, title, output_file):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=labels,
        cmap="hsv",  # 24개 클러스터에 맞는 색상 맵
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("TSNE Dim 1")
    plt.ylabel("TSNE Dim 2")
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    plt.close()

# 시각화 저장 디렉토리 설정
output_folder = "clustering_visualizations"
os.makedirs(output_folder, exist_ok=True)

# KMeans 시각화
plot_clustering(
    kmeans_tsne,
    kmeans_labels,
    "KMeans Clustering Visualization",
    os.path.join(output_folder, f"kmeans_clustering_{BIN_NUM}.png"),
)

# Manifold 시각화
plot_clustering(
    manifold_tsne,
    manifold_labels,
    "Grassmannian Clustering Visualization",
    os.path.join(output_folder, f"manifold_clustering_{BIN_NUM}.png"),
)
print(f"All visualizations saved in {output_folder}")
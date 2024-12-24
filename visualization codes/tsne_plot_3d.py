import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from models.action_ae.discretizers.k_means import KMeansDiscretizer
from models.action_ae.discretizers.manifold import ManifoldDiscretizer
from models.action_ae.discretizers.Grassmannian import GrassmannianDiscretizer

# 데이터 로드
path = "/root/bet/bet_data_release/blockpush/multimodal_push_actions.npy"
actions = torch.tensor(np.load(path), dtype=torch.float32)
print(f"Loaded actions shape: {actions.shape}")

# TSNE 시각화를 위한 데이터 준비 함수
def prepare_tsne_data_3d(actions, discretizer, name, n_components=3):
    flattened_actions = actions.view(-1, actions.shape[-1])
    labels = discretizer.encode_into_latent(flattened_actions).view(-1).cpu().numpy()
    tsne = TSNE(n_components=n_components, random_state=42, n_jobs=-1, verbose=1)
    tsne_result = tsne.fit_transform(flattened_actions.cpu().numpy())
    return tsne_result, labels, name

# 시각화 함수 (3D subplot)
def plot_3d_clustering_subplot(tsne_results, labels_list, names, k, output_file):
    num_methods = len(names)
    fig = plt.figure(figsize=(8 * num_methods, 8))
    
    for i, (tsne_result, labels, name) in enumerate(zip(tsne_results, labels_list, names)):
        ax = fig.add_subplot(1, num_methods, i + 1, projection='3d')
        scatter = ax.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            tsne_result[:, 2],
            c=labels,
            cmap="hsv",
            alpha=0.6,
        )
        ax.set_title(f"{name} (k={k})", fontsize=14)
        ax.set_xlabel("TSNE Dim 1")
        ax.set_ylabel("TSNE Dim 2")
        ax.set_zlabel("TSNE Dim 3")
        fig.colorbar(scatter, ax=ax, label="Cluster")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved 3D subplot to {output_file}")
    plt.close()

# k 값 목록 및 결과 저장 디렉토리
k_values = [6, 12, 24, 48, 96]
output_folder_3d = "clustering_visualizations_3d"
os.makedirs(output_folder_3d, exist_ok=True)

# 클러스터링 수행 및 시각화
names = ["KMeans", "Manifold", "Geodesic"]
for k in k_values:
    print(f"Processing 3D TSNE for k={k}...")
    
    # KMeans 클러스터링
    kmeans_discretizer = KMeansDiscretizer(action_dim=actions.shape[-1], num_bins=k, device="cpu")
    kmeans_discretizer.fit_discretizer(actions)
    kmeans_tsne_3d, kmeans_labels, _ = prepare_tsne_data_3d(actions, kmeans_discretizer, "KMeans", n_components=3)
    
    # Manifold 클러스터링
    manifold_discretizer = ManifoldDiscretizer(action_dim=actions.shape[-1], num_bins=k, device="cuda")
    manifold_discretizer.fit_discretizer(actions)
    manifold_tsne_3d, manifold_labels, _ = prepare_tsne_data_3d(actions, manifold_discretizer, "Manifold", n_components=3)
    
    # Geodesic 클러스터링
    geodesic_discretizer = GrassmannianDiscretizer(action_dim=actions.shape[-1], num_bins=k, device="cuda")
    geodesic_discretizer.fit_discretizer(actions)
    geodesic_tsne_3d, geodesic_labels, _ = prepare_tsne_data_3d(actions, geodesic_discretizer, "Geodesic", n_components=3)
    
    # 3D TSNE 시각화 subplot 저장
    output_file = os.path.join(output_folder_3d, f"clustering_3d_k{k}.png")
    plot_3d_clustering_subplot(
        [kmeans_tsne_3d, manifold_tsne_3d, geodesic_tsne_3d],
        [kmeans_labels, manifold_labels, geodesic_labels],
        names,
        k,
        output_file,
    )

print(f"All 3D visualizations saved in {output_folder_3d}")
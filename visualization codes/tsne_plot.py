import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models.action_ae.discretizers.k_means import KMeansDiscretizer
from models.action_ae.discretizers.manifold import ManifoldDiscretizer
from models.action_ae.discretizers.Grassmannian import GrassmannianDiscretizer

# 데이터 로드
path = "/root/bet/bet_data_release/blockpush/multimodal_push_actions.npy"
actions = torch.tensor(np.load(path), dtype=torch.float32)
print(f"Loaded actions shape: {actions.shape}")

# TSNE 시각화를 위한 데이터 준비 함수
def prepare_tsne_data(actions, discretizer, name):
    flattened_actions = actions.view(-1, actions.shape[-1])
    labels = discretizer.encode_into_latent(flattened_actions).view(-1).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, verbose=1)
    tsne_result = tsne.fit_transform(flattened_actions.cpu().numpy())
    return tsne_result, labels, name

# 시각화 함수 (subplot 포함)
def plot_clustering_results(tsne_results, labels_list, names, k_values, output_file):
    num_methods = len(names)
    fig, axs = plt.subplots(len(k_values), num_methods, figsize=(8 * num_methods, 6 * len(k_values)))

    for i, k in enumerate(k_values):
        for j, (tsne_result, labels, name) in enumerate(zip(tsne_results, labels_list, names)):
            ax = axs[i, j] if len(k_values) > 1 else axs[j]
            scatter = ax.scatter(
                tsne_result[i][:, 0],
                tsne_result[i][:, 1],
                c=labels[i],
                cmap="hsv",
                alpha=0.6,
            )
            ax.set_title(f"{name} (k={k})", fontsize=14)
            ax.set_xlabel("TSNE Dim 1")
            ax.set_ylabel("TSNE Dim 2")
            ax.grid(True)
            if j == num_methods - 1:
                fig.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved combined plot to {output_file}")
    plt.close()

# k 값 목록 및 결과 저장 디렉토리
k_values = [6, 12, 24, 48, 96]
output_folder = "clustering_visualizations"
os.makedirs(output_folder, exist_ok=True)

# 클러스터링 수행 및 시각화
tsne_results = []
labels_list = []
names = ["KMeans", "Manifold", "Geodesic"]
# 클러스터링 및 시각화를 개별적으로 저장
for k in k_values:
    print(f"Processing for k={k}...")

    try:
        # KMeans 클러스터링
        kmeans_discretizer = KMeansDiscretizer(action_dim=actions.shape[-1], num_bins=k, device="cpu")
        kmeans_discretizer.fit_discretizer(actions)
        kmeans_tsne, kmeans_labels, _ = prepare_tsne_data(actions, kmeans_discretizer, "KMeans")

        # Manifold 클러스터링
        manifold_discretizer = ManifoldDiscretizer(action_dim=actions.shape[-1], num_bins=k, device="cuda")
        manifold_discretizer.fit_discretizer(actions)
        manifold_tsne, manifold_labels, _ = prepare_tsne_data(actions, manifold_discretizer, "Manifold")

        # Geodesic 클러스터링
        geodesic_discretizer = GrassmannianDiscretizer(action_dim=actions.shape[-1], num_bins=k, device="cuda")
        geodesic_discretizer.fit_discretizer(actions)
        geodesic_tsne, geodesic_labels, _ = prepare_tsne_data(actions, geodesic_discretizer, "Geodesic")

        # TSNE 결과 저장
        tsne_results = [kmeans_tsne, manifold_tsne, geodesic_tsne]
        labels_list = [kmeans_labels, manifold_labels, geodesic_labels]

        # 개별 결과 시각화
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        for idx, (tsne_result, labels, name) in enumerate(zip(tsne_results, labels_list, names)):
            scatter = axs[idx].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="hsv", alpha=0.6)
            axs[idx].set_title(f"{name} (k={k})", fontsize=14)
            axs[idx].set_xlabel("TSNE Dim 1")
            axs[idx].set_ylabel("TSNE Dim 2")
            axs[idx].grid(True)
            fig.colorbar(scatter, ax=axs[idx], label="Cluster")
        
        output_file = os.path.join(output_folder, f"clustering_results_k{k}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Saved visualization for k={k} to {output_file}")
        plt.close()
    except Exception as e:
        print(f"Error occurred for k={k}: {e}")
        
# 결과 시각화
output_file = os.path.join(output_folder, "combined_clustering_results.png")
plot_clustering_results(tsne_results, labels_list, names, k_values, output_file)

print(f"All visualizations saved in {output_folder}")
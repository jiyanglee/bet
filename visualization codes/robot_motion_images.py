import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from models.action_ae.discretizers.manifold import ManifoldDiscretizer
from models.action_ae.discretizers.k_means import KMeansDiscretizer  # KMeansDiscretizer 클래스 가져오기

# 데이터 로드
path = "bet_data_release/blockpush"
actions = torch.tensor(np.load(os.path.join(path, "multimodal_push_actions.npy")), dtype=torch.float32)

# Geodesic 디스크리타이저
num_bins = 36
action_dim = actions.shape[-1]
geodesic_discretizer = ManifoldDiscretizer(action_dim=action_dim, num_bins=36, device="cuda")
geodesic_discretizer.fit_discretizer(actions)

# Euclidean(KMeans) 디스크리타이저
euclidean_discretizer = KMeansDiscretizer(action_dim=action_dim, num_bins=24, device="cuda")
euclidean_discretizer.fit_discretizer(actions)

# Euclidean 클러스터 센터를 CUDA로 이동
euclidean_discretizer.bin_centers = euclidean_discretizer.bin_centers.to("cuda")

# 클러스터링
geodesic_clusters, euclidean_clusters = [], []
for i in tqdm.trange(actions.shape[0], desc="Clustering motions"):
    sequence = actions[i].to("cuda")  # sequence를 CUDA로 이동
    geodesic_clusters.append(geodesic_discretizer.encode_into_latent(sequence).cpu().numpy())
    euclidean_clusters.append(euclidean_discretizer.encode_into_latent(sequence).cpu().numpy())

geodesic_clusters = np.array(geodesic_clusters)
euclidean_clusters = np.array(euclidean_clusters)

# 클러스터 색상 생성 함수
def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(num_clusters)]
    return colors

# 시각화 및 저장 함수
def save_comparison_image(sequence_index, actions, geodesic_clusters, euclidean_clusters, output_folder, num_bins):
    sequence = actions[sequence_index]
    geodesic_assignments = geodesic_clusters[sequence_index].flatten()
    euclidean_assignments = euclidean_clusters[sequence_index].flatten()
    
    # 클러스터 색상
    cluster_colors = generate_cluster_colors(num_bins)

    # 시각화
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Clustered Motion Comparison (Sequence {sequence_index})", fontsize=16)

    # Original Motion
    axs[0].plot(sequence[:, 0].cpu().numpy(),
                sequence[:, 1].cpu().numpy(),
                linestyle="--", color="gray", label="Original Motion", alpha=0.5)
    axs[0].set_title("Original Motion")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    # Clustered Motion (Geodesic)
    for i in range(len(sequence) - 1):
        cluster_id = geodesic_assignments[i]
        color = cluster_colors[cluster_id]
        axs[1].plot(sequence[i:i+2, 0].cpu().numpy(),
                    sequence[i:i+2, 1].cpu().numpy(),
                    color=color, linewidth=2, alpha=0.8)
    axs[1].set_title("Clustered Motion (Geodesic)")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")

    # Clustered Motion (Euclidean)
    for i in range(len(sequence) - 1):
        cluster_id = euclidean_assignments[i]
        color = cluster_colors[cluster_id]
        axs[2].plot(sequence[i:i+2, 0].cpu().numpy(),
                    sequence[i:i+2, 1].cpu().numpy(),
                    color=color, linewidth=2, alpha=0.8, linestyle=":")
    axs[2].set_title("Clustered Motion (Euclidean)")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")

    # 이미지 저장
    image_path = os.path.join(output_folder, f"comparison_sequence_{sequence_index}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(image_path)
    print(f"Saved comparison image: {image_path}")
    plt.close(fig)

# 결과 저장 디렉토리 생성
comparison_output_folder = "2d_comparison_images"
os.makedirs(comparison_output_folder, exist_ok=True)

# 모든 sequence 비교 이미지 저장
for sequence_index in tqdm.trange(actions.shape[0], desc="Saving comparison images"):
    save_comparison_image(sequence_index, actions, geodesic_clusters, euclidean_clusters, comparison_output_folder, num_bins)

print(f"All images saved in folder: {comparison_output_folder}")
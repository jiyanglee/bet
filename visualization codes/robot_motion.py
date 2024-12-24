import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

# 로봇 모션 데이터를 시뮬레이션하거나 불러오는 함수 (여기서는 랜덤 데이터 사용)
def generate_motion_data(num_frames=50, dim=3):
    # 랜덤 3D 모션 데이터 생성
    data = np.cumsum(np.random.randn(num_frames, dim), axis=0)  # 누적합으로 움직임 시뮬레이션
    return torch.tensor(data, dtype=torch.float32)

# 거리 계산 함수
def calculate_distances(data):
    num_frames = data.shape[0]
    euclidean_distances = []
    geodesic_distances = []

    for i in range(num_frames - 1):
        # Euclidean 거리 계산
        euclidean_dist = torch.norm(data[i + 1] - data[i], p=2).item()
        euclidean_distances.append(euclidean_dist)

        # Geodesic 거리 계산 (단순히 방향성 유지)
        # 실제 Geodesic은 다양체가 필요하지만 여기선 방향 기반 근사
        dot_product = torch.dot(data[i], data[i + 1]) / (
            torch.norm(data[i]) * torch.norm(data[i + 1])
        )
        geodesic_dist = torch.acos(torch.clamp(dot_product, -1.0, 1.0)).item()
        geodesic_distances.append(geodesic_dist)

    return euclidean_distances, geodesic_distances
def visualize_motion_and_distances(motion_data, euclidean_dists, geodesic_dists):
    # torch.Tensor를 NumPy 배열로 변환
    motion_data_np = motion_data.cpu().numpy() if isinstance(motion_data, torch.Tensor) else motion_data
    euclidean_dists_np = np.array(euclidean_dists)
    geodesic_dists_np = np.array(geodesic_dists)

    # 3D 모션 경로 시각화
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(
        motion_data_np[:, 0], 
        motion_data_np[:, 1], 
        motion_data_np[:, 2], 
        label="Robot Motion"
    )
    ax.set_title("Robot Motion in 3D")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()

    # 거리 시각화
    ax2 = fig.add_subplot(122)
    ax2.plot(
        range(len(euclidean_dists_np)), 
        euclidean_dists_np, 
        label="Euclidean Distance", 
        color="blue"
    )
    ax2.plot(
        range(len(geodesic_dists_np)), 
        geodesic_dists_np, 
        label="Geodesic Distance", 
        color="red"
    )
    ax2.set_title("Distances Between Frames")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Distance")
    ax2.legend()

    # 파일로 저장
    plt.savefig("robot_motion_and_distances.png")
    print("Graph saved as 'robot_motion_and_distances.png'")
    
# 메인 프로그램
if __name__ == "__main__":
    # 로봇 모션 데이터 생성
    motion_data = generate_motion_data(num_frames=50, dim=3)

    # 거리 계산
    euclidean_dists, geodesic_dists = calculate_distances(motion_data)

    # 모션과 거리 시각화
    visualize_motion_and_distances(motion_data, euclidean_dists, geodesic_dists)

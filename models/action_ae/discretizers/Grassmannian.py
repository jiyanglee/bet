import torch
import numpy as np
import tqdm
from typing import Optional, Tuple, Union
from models.action_ae.discretizers.base import AbstractDiscretizer


class GrassmannianDiscretizer(AbstractDiscretizer):
    """
    Discretizer for action representations using Grassmannian Geodesic Distance.
    Optimized for GPU and vectorized operations.
    """

    def __init__(
        self,
        action_dim: int,
        num_bins: int = 100,
        device: Union[str, torch.device] = "cuda",  # Default to GPU
        predict_offsets: bool = False,
        k_neighbors: int = 10,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets
        self.k_neighbors = k_neighbors

    def fit_discretizer(self, input_actions: torch.Tensor) -> None:
        assert (
            self.action_dim == input_actions.shape[-1]
        ), f"Input action dimension {self.action_dim} does not match fitted model {input_actions.shape[-1]}"

        flattened_actions = input_actions.view(-1, self.action_dim).to(self.device)
        cluster_centers = self._manifold_kmeans(flattened_actions, self.n_bins)
        self.bin_centers = cluster_centers.to(self.device)

    @property
    def suggested_actions(self) -> torch.Tensor:
        return self.bin_centers

    def _geodesic_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Geodesic distances on the Grassmannian.
        """
        # Compute projection matrices
        proj_x = torch.einsum("bi,bj->bij", x, x)  # Projection matrix for x
        proj_y = torch.einsum("bi,bj->bij", y, y)  # Projection matrix for y

        # Ensure proj_x and proj_y have compatible batch sizes for pairwise computation
        proj_x = proj_x.unsqueeze(1)  # Shape: (batch_size_x, 1, dim, dim)
        proj_y = proj_y.unsqueeze(0)  # Shape: (1, batch_size_y, dim, dim)

        # Compute the principal angles using singular value decomposition (SVD)
        svd_values = torch.linalg.svdvals(
            torch.matmul(proj_x, proj_y)
        )  # Pairwise computation

        # Clamp values to avoid numerical issues
        svd_values = torch.clamp(svd_values, -1.0, 1.0)

        # Compute the geodesic distance
        principal_angles = torch.acos(svd_values)
        geodesic_distance = torch.sqrt(torch.sum(principal_angles**2, dim=-1))

        return geodesic_distance

    def _manifold_kmeans(
        self, x: torch.Tensor, ncluster: int, niter: int = 50
    ) -> torch.Tensor:
        """
        Perform optimized k-means clustering using Grassmannian geodesic distance.
        """
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]]  # Initialize clusters randomly
        c = c.to(self.device)

        pbar = tqdm.trange(niter)
        pbar.set_description("Grassmannian Geodesic K-means clustering")
        for i in pbar:
            # Compute distances in a vectorized manner
            distances = self._geodesic_metric(x, c)

            # Assign points to the closest cluster center
            a = distances.argmin(dim=1)

            # Update cluster centers to be the Grassmannian mean of assigned points
            new_centers = []
            for k in range(ncluster):
                assigned_points = x[a == k]
                if len(assigned_points) > 0:
                    mean = assigned_points.mean(dim=0)
                    norm_mean = torch.norm(mean, p=2, dim=-1, keepdim=True)
                    new_centers.append(mean / norm_mean)  # Normalize to manifold
                else:
                    new_centers.append(c[k])
            c = torch.stack(new_centers)
        return c

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert (
            input_action.shape[-1] == self.action_dim
        ), "Input action dimension does not match fitted model"

        flattened_actions = input_action.view(-1, self.action_dim).to(self.device)
        distances = self._geodesic_metric(flattened_actions, self.bin_centers)
        closest_cluster_center = distances.argmin(dim=1)
        discretized_action = closest_cluster_center.view(input_action.shape[:-1] + (1,))

        if self.predict_offsets:
            reconstructed_action = self.decode_actions(discretized_action)
            offsets = input_action - reconstructed_action
            return (discretized_action, offsets)
        else:
            return discretized_action

    def decode_actions(
        self,
        latent_action_batch: torch.Tensor,
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        offsets = None
        if type(latent_action_batch) == tuple:
            latent_action_batch, offsets = latent_action_batch
        closest_cluster_center = self.bin_centers[latent_action_batch]
        reconstructed_action = closest_cluster_center.view(
            latent_action_batch.shape[:-1] + (self.action_dim,)
        )
        if offsets is not None:
            reconstructed_action += offsets
        return reconstructed_action

    @property
    def discretized_space(self) -> int:
        return self.n_bins

    @property
    def latent_dim(self) -> int:
        return 1

    @property
    def num_latents(self) -> int:
        return self.n_bins

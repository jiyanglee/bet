import torch
import numpy as np
import tqdm
from typing import Optional, Tuple, Union
from models.action_ae.discretizers.base import AbstractDiscretizer


class ProjectionBasedDiscretizer(AbstractDiscretizer):
    """
    Discretizer for action representations using Grassmannian Projection Distance.
    Optimized for GPU and vectorized operations.
    """

    def __init__(
        self,
        action_dim: int,
        num_bins: int = 100,
        device: Union[str, torch.device] = "cuda",
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
        cluster_centers = self._projection_kmeans(flattened_actions, self.n_bins)
        self.bin_centers = cluster_centers.to(self.device)

    @property
    def suggested_actions(self) -> torch.Tensor:
        return self.bin_centers

    def _projection_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Grassmannian projection distances using Frobenius norm.
        This metric captures differences between projection matrices of input points.
        """
        proj_x = torch.einsum("bi,bj->bij", x, x)
        proj_y = torch.einsum("bi,bj->bij", y, y)
        distance = torch.norm(
            proj_x[:, None, :, :] - proj_y[None, :, :, :], dim=(-2, -1), p="fro"
        )
        return distance

    def _projection_kmeans(
        self, x: torch.Tensor, ncluster: int, niter: int = 50
    ) -> torch.Tensor:
        """
        Perform k-means clustering using Grassmannian projection distances.
        Cluster centers are updated as normalized means on the Grassmannian manifold.
        """
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]].to(self.device)

        pbar = tqdm.trange(niter)
        pbar.set_description("Grassmannian Projection K-means clustering")
        for i in pbar:
            distances = self._projection_metric(x, c)
            a = distances.argmin(dim=1)

            new_centers = []
            for k in range(ncluster):
                assigned_points = x[a == k]
                if len(assigned_points) > 0:
                    mean = assigned_points.mean(dim=0)
                    norm_mean = torch.norm(mean, p=2, dim=-1, keepdim=True)
                    new_centers.append(mean / norm_mean)
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
        distances = self._projection_metric(flattened_actions, self.bin_centers)
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

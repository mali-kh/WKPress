# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from kvpress.presses.base_press import BasePress
logger = logging.getLogger(__name__)
@dataclass
class WKPress(BasePress):
    """
    Whitened k-center KV cache compression using Gonzalez farthest-point sampling.
    This method selects K tokens so every key direction is close (in whitened space)
    to at least one kept token. It covers the geometry of K without using attention.
    Algorithm:
    1. Compute G = K^T @ K (Gram matrix)
    2. Add ridge regularization: Gt = G + lambda * I
    3. Cholesky decomposition: Gt = L @ L^T
    4. Whiten keys: B = K @ inv(L^T)
    5. Apply Gonzalez farthest-point sampling to select k-center indices
    Parameters
    ----------
    compression_ratio : float, default=0.0
    Fraction of key-value pairs to remove during compression.
    lambda_reg : float, default=1e-6
    Ridge regularization parameter for numerical stability.
    """
    compression_ratio: float = 0.0
    lambda_reg: float = 1e-4
    
    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert self.lambda_reg > 0, "Lambda regularization must be positive"
    
    def _compute_cholesky_and_whiten(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute Cholesky decomposition and whiten the key vectors.
        Parameters
        ----------
        keys : torch.Tensor
        Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        Returns
        -------
        torch.Tensor
        Whitened keys B with shape (batch_size, num_kv_heads, seq_len, head_dim).
        """
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        # Convert to float32 for numerical stability if needed
        original_dtype = keys.dtype
        needs_cast = keys.dtype in [torch.bfloat16, torch.float16]
        if needs_cast:
            keys = keys.float()
        # Ensure contiguous memory layout for faster matmul
        keys = keys.contiguous()
        keys_T = keys.transpose(-2, -1).contiguous()
        # Vectorized computation: Gram matrix G = K^T @ K for all heads at once
        G = torch.matmul(keys_T, keys) # (batch, heads, dim, dim)
        # Add ridge regularization (in-place for efficiency)
        eye = torch.eye(head_dim, device=G.device, dtype=G.dtype)
        G.add_(eye.unsqueeze(0).unsqueeze(0), alpha=self.lambda_reg)
        # Cholesky decomposition for all heads at once
        try:
            L = torch.linalg.cholesky(G) # (batch, heads, dim, dim)
        except torch.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails (rare with adequate lambda_reg)
            U, S, Vh = torch.linalg.svd(G)
            S_sqrt = torch.sqrt(S.clamp(min=0) + self.lambda_reg)
            L = U @ torch.diag_embed(S_sqrt) @ Vh
        # Whiten: B = K @ inv(L^T)
        # Solve L^T @ X^T = K^T for X^T, where X is whitened_keys
        L_T = L.transpose(-2, -1).contiguous()
        whitened_keys = torch.linalg.solve(L_T, keys_T).transpose(-2, -1).contiguous()
        # Convert back to original dtype if needed
        if needs_cast:
            whitened_keys = whitened_keys.to(original_dtype)
        return whitened_keys
    
    def _rowwise_squared_distance(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute squared distances between each row of X and vector y.
        Parameters
        ----------
        X : torch.Tensor
        Matrix with shape (n, d).
        y : torch.Tensor
        Vector with shape (d,).
        Returns
        -------
        torch.Tensor
        Squared distances with shape (n,).
        """
        # Efficient computation: ||X - y||^2 = ||X||^2 + ||y||^2 - 2*X@y
        X_norm_sq = torch.sum(X**2, dim=-1) # (n,)
        y_norm_sq = torch.sum(y**2) # scalar
        dot_product = torch.sum(X * y, dim=-1) # (n,)
        return X_norm_sq + y_norm_sq - 2 * dot_product
    
    def _gonzalez_farthest_point_sampling(self, whitened_keys: torch.Tensor, keep_K: int) -> torch.Tensor:
        """
        Apply Gonzalez farthest-point sampling to select k-center indices.
        Vectorized across batch and heads for efficiency.
        Parameters
        ----------
        whitened_keys : torch.Tensor
        Whitened key vectors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        keep_K : int
        Number of tokens to keep.
        Returns
        -------
        torch.Tensor
        Indices of selected tokens with shape (batch_size, num_kv_heads, keep_K).
        """
        batch_size, num_kv_heads, seq_len, head_dim = whitened_keys.shape
        # Convert to float32 for numerical stability if needed
        if whitened_keys.dtype in [torch.bfloat16, torch.float16]:
            whitened_keys = whitened_keys.float()
        # Reshape to combine batch and heads: (batch * heads, seq_len, head_dim)
        B = whitened_keys.view(-1, seq_len, head_dim).contiguous()
        num_groups = B.shape[0] # batch_size * num_kv_heads
        # Pre-allocate all tensors to avoid repeated allocations
        selected_indices = torch.zeros(num_groups, keep_K, dtype=torch.long, device=B.device)
        group_arange = torch.arange(num_groups, device=B.device) # Reuse this
        # Pre-compute all row norms once (won't change)
        B_norms = torch.sum(B * B, dim=-1) # (num_groups, seq_len) - in-place computation
        # Initialize with tokens having maximum norm
        start_indices = torch.argmax(torch.sqrt(B_norms), dim=-1) # (num_groups,)
        selected_indices[:, 0] = start_indices
        # Initialize distances to first center
        first_centers = B[group_arange, start_indices] # (num_groups, head_dim)
        center_norms = torch.sum(first_centers * first_centers, dim=-1, keepdim=True) # (num_groups, 1)
        # Use bmm with pre-allocated output buffer
        dot_products = torch.bmm(B, first_centers.unsqueeze(-1)).squeeze_(-1) # (num_groups, seq_len)
        distances = B_norms + center_norms - 2 * dot_products # (num_groups, seq_len)
        # Pre-allocate temporary tensors for the loop
        new_center_norms = torch.empty(num_groups, 1, dtype=B.dtype, device=B.device)
        new_dot_products = torch.empty(num_groups, seq_len, dtype=B.dtype, device=B.device)
        new_distances = torch.empty(num_groups, seq_len, dtype=B.dtype, device=B.device)
        # Iteratively select farthest points
        for k in range(1, keep_K):
            # Find the farthest point from current centers
            next_indices = torch.argmax(distances, dim=-1) # (num_groups,)
            selected_indices[:, k] = next_indices
            # Get the newly selected centers (reuse group_arange)
            new_centers = B[group_arange, next_indices] # (num_groups, head_dim)
            # Compute distances to new centers using pre-allocated tensors
            torch.sum(new_centers * new_centers, dim=-1, keepdim=True, out=new_center_norms)
            torch.bmm(B, new_centers.unsqueeze(-1), out=new_dot_products.unsqueeze(-1))
            # new_distances = B_norms + new_center_norms - 2 * new_dot_products
            new_distances.copy_(B_norms).add_(new_center_norms).add_(new_dot_products, alpha=-2.0)
            # Update distances to nearest chosen center (in-place)
            torch.minimum(distances, new_distances, out=distances)
        # Reshape back to original batch structure
        return selected_indices.view(batch_size, num_kv_heads, keep_K)
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress keys and values using whitened k-center selection.
        Parameters
        ----------
        module : nn.Module
        The transformer attention layer where compression is applied.
        hidden_states : torch.Tensor
        Hidden states with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
        Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
        Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
        Attention weights (not used in this method).
        kwargs : dict
        Additional arguments from the forward pass.
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        Compressed keys and values tensors.
        """
        if self.compression_ratio == 0:
            return keys, values
        # Calculate number of tokens to keep
        seq_len = keys.shape[2]
        keep_K = int(seq_len * (1 - self.compression_ratio))
        if keep_K <= 0:
            # Keep at least one token
            keep_K = 1
        elif keep_K >= seq_len:
            # Keep all tokens
            return keys, values
        # Step 1: Whiten the key vectors
        whitened_keys = self._compute_cholesky_and_whiten(keys)
        # Step 2: Apply Gonzalez farthest-point sampling
        selected_indices = self._gonzalez_farthest_point_sampling(whitened_keys, keep_K)
        # Step 3: Select the corresponding keys and values
        # Expand indices for gathering: (batch_size, num_kv_heads, keep_K, head_dim)
        indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        # Gather compressed keys and values
        compressed_keys = keys.gather(2, indices_expanded).contiguous()
        compressed_values = values.gather(2, indices_expanded).contiguous()
        return compressed_keys, compressed_values


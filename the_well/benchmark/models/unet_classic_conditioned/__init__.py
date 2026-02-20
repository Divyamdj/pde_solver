import torch
import torch.nn as nn
import logging

from the_well.benchmark.models.unet_classic import UNetClassic

logger = logging.getLogger("the_well")


class UNetClassicConditioned(nn.Module):
    """
    UNetClassic conditioned on PDE embedding by projecting embedding to spatial maps
    and concatenating to the input channels.
    """

    def __init__(
        self,
        dim_in: int,                 # input channels (already includes Ti*C)
        dim_out: int,                # output channels (usually To*C, To=1 => C)
        n_spatial_dims: int = 2,
        spatial_resolution: tuple[int, ...] = (128, 384),
        init_features: int = 48,
        emb_dim: int = 384,
        cond_channels: int = 16,
        n_steps_output: int = 1,     # for reshaping back to (B, To, H, W, C)
    ):
        super().__init__()

        assert n_spatial_dims == 2, "This conditioned UNet currently assumes 2D."
        self.cond_channels = cond_channels
        self.dim_out = dim_out
        self.n_steps_output = n_steps_output
        self.dim_in = dim_in
        self._shape_logged = False  # Track if we've already logged shapes

        # Project PDE text embedding -> cond_channels
        self.embed_proj = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cond_channels),
        )

        # Base UNet
        self.base = UNetClassic(
            dim_in=dim_in + cond_channels,
            dim_out=dim_out,
            n_spatial_dims=n_spatial_dims,
            spatial_resolution=spatial_resolution,
            init_features=init_features,
        )

    def forward(self, x, pde_embedding=None):
        if pde_embedding is None:
            raise ValueError("pde_embedding is required")

        # Log input shapes on first forward pass
        if not self._shape_logged:
            logger.info(f"UNetClassicConditioned input shapes: x={x.shape}, pde_embedding={pde_embedding.shape}")
            logger.info(f"Expected input channels: {self.dim_in}, Output channels: {self.dim_out}")
            self._shape_logged = True

        pde_embedding = pde_embedding.to(x.device)

        if pde_embedding.ndim == 1:
            pde_embedding = pde_embedding.unsqueeze(0).repeat(x.shape[0], 1)

        B, C, H, W = x.shape

        cond = self.embed_proj(pde_embedding)  # (B, cond_channels)
        cond = cond[:, :, None, None].expand(B, self.cond_channels, H, W)

        x = torch.cat([x, cond], dim=1)

        y = self.base(x)  # (B, To*C, H, W)
        return y

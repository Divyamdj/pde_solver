# the_well/data/pde_text_embedder.py

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

DESCRIPTIONS = [
    "Turbulent radiative layer in two spatial dimensions governed by compressible Navier–Stokes equations coupled with thermal energy transport. The system includes nonlinear advection, viscous diffusion, and buoyancy forcing driven by vertical temperature gradients. Lower boundary injects heat, upper boundary removes heat through radiative cooling, horizontal boundaries are periodic. Flow regime is convection driven and turbulence dominated. The solution develops multi-scale vortices, thermal plumes, and chaotic velocity structures. The predicted state should exhibit intermittent turbulent mixing and filamentary temperature fields.",
    "Two-dimensional active matter system governed by nonlinear hydrodynamic equations with self-propulsion forcing and diffusive regularization. The velocity field includes advective transport and active energy injection at small scales. Periodic boundary conditions in both spatial directions. The system is far from equilibrium and activity dominated. The solution develops spontaneous vortices, traveling bands, and long-range collective motion. The predicted state should exhibit coherent swirling structures and dynamically forming clusters.",
]

@dataclass
class PDETextEmbedder:
    """
    Computes a text embedding for each dataset name using a frozen sentence-transformer.
    Caches embeddings to disk so you don't recompute every run.

    The embedder cycles through a list of descriptive sentences so that each
    generated text incorporates exactly one description; subsequent calls use the
    next sentence in the list (wrapping around) to avoid repeating all
    descriptions on every example.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_path: str = "pde_text_embeddings.pt"
    device: str = "cpu"

    def __post_init__(self):
        self._model = None
        self._cache: Dict[str, torch.Tensor] = {}
        self._desc_idx = 0  # index into DESCRIPTIONS for round-robin

        if os.path.exists(self.cache_path):
            try:
                self._cache = torch.load(self.cache_path, map_location="cpu")
            except Exception:
                self._cache = {}

    def _lazy_load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)

# descriptive sentences used during PDE embedding training


    def get_text_for_dataset(
        self,
        dataset_name: str,
        field_names: Optional[list[str]] = None,
        bc_types: Optional[list[str]] = None,
        spatial_res: Optional[tuple[int, ...]] = None,
        n_spatial_dims: Optional[int] = None,
    ) -> str:
        # Keep it stable + consistent
        text = f"Dataset: {dataset_name}. Task: predict future state from past states."
        if n_spatial_dims is not None:
            text += f" Spatial dims: {n_spatial_dims}."
        if spatial_res is not None:
            text += f" Resolution: {spatial_res}."
        if field_names:
            text += " Fields: " + ", ".join(field_names) + "."
        if bc_types:
            text += " Boundary conditions: " + ", ".join(bc_types) + "."
        # append a single training description (cycle through list)
        if len(DESCRIPTIONS) > 0:
            desc = DESCRIPTIONS[self._desc_idx % len(DESCRIPTIONS)]
            self._desc_idx += 1
            text += " " + desc

        return text

    @torch.no_grad()
    def embed(self, key: str, text: str) -> torch.Tensor:
        """
        Returns a CPU float32 embedding tensor of shape (D,).
        """
        if key in self._cache:
            return self._cache[key]

        self._lazy_load_model()
        emb = self._model.encode(text, convert_to_tensor=True)
        emb = emb.detach().float().cpu()

        self._cache[key] = emb
        torch.save(self._cache, self.cache_path)
        return emb

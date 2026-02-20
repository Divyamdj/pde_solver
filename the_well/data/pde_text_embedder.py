# the_well/data/pde_text_embedder.py

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

DESCRIPTIONS = [
    "This system models 2D compressible turbulence in a radiatively cooling layer.",
    "The equations describe conservation of mass through the continuity equation.",
    "This PDE set represents momentum evolution influenced by pressure gradients in two dimensions.",
    "The system captures energy evolution under advection and radiative cooling.",
    "The equations model 2D turbulent flow where energy decreases over a characteristic cooling time.",
    "This model describes velocity evolution in a compressible medium with pressure forces.",
    "The PDEs represent total energy as the sum of internal energy and kinetic energy.",
    "The system describes how radiative cooling removes energy from the turbulent layer.",
    "The equations capture the coupling between density, velocity, and pressure in a 2D layer.",
    "This model describes the advection of momentum by the velocity field itself.",
    "The PDEs represent how pressure gradients drive acceleration in the turbulent flow.",
    "The system models energy redistribution due to fluid motion and compressional work.",
    "The equations describe how density fluctuations are transported by the velocity field.",
    "This model captures radiative energy loss proportional to the total energy over the cooling time.",
    "The PDEs represent nonlinear advection of momentum and energy in a compressible layer.",
    "The system describes turbulence where the adiabatic index γ governs the pressure–energy relationship.",
    "The equations model the effect of compressibility on 2D velocity and density fields.",
    "This model captures interactions between kinetic energy, pressure, and radiative cooling.",
    "The PDEs describe how pressure evolves from density and internal energy.",
    "The system represents compressible turbulence damped by radiative cooling.",
    "The equations model 2D flow structures shaped by nonlinear advection and pressure forces.",
    "This model describes energy dissipation in a turbulent layer through radiative losses.",
    "The PDEs represent the effect of velocity divergence on density evolution.",
    "The system captures compressible turbulent motions in a 2D radiative medium.",
    "The equations describe how radiative cooling affects the dynamics of both momentum and energy.",
    "This model represents the evolution of total energy under pressure work and cooling.",
    "The PDEs describe how velocity gradients influence momentum transport.",
    "The system models turbulent flow where cooling reduces internal energy over time.",
    "The equations capture the balance between pressure acceleration and radiative energy loss.",
    "This model describes 2D compressible turbulence in a layer with fixed γ = 5/3.",
    "The PDEs represent energy evolution influenced by advection and cooling simultaneously.",
    "The system describes momentum advection coupled to pressure gradients in a radiatively active layer.",
    "The equations model turbulent energy redistribution in a compressible 2D medium.",
    "This model captures how density and velocity fluctuations interact with radiative cooling.",
    "The PDEs describe how compressibility modifies energy transport in 2D turbulence.",
    "The system represents total energy evolution including internal, kinetic, and cooling contributions.",
    "The equations model pressure-driven acceleration in a radiatively cooling turbulent layer.",
    "This model describes how cooling time determines the rate of energy loss.",
    "The PDEs capture 2D turbulent motions where velocity advects both momentum and energy.",
    "The system represents compressible hydrodynamics with pressure forces and radiative damping in a 2D layer."
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

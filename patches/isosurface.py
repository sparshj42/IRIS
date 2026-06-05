from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from skimage.measure import marching_cubes as skimage_marching_cubes

def marching_cubes(level, threshold):
    level_np = level.detach().cpu().numpy()
    v_pos, t_pos_idx, _, _ = skimage_marching_cubes(level_np, threshold)
    v_pos = torch.from_numpy(v_pos.copy()).float()
    t_pos_idx = torch.from_numpy(t_pos_idx.copy()).long()
    return v_pos, t_pos_idx

class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)
    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError

class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos, t_pos_idx

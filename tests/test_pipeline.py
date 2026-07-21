"""Basic invariants for the cymatics plane → line pipeline."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.pipeline import run_pipeline


def test_default_pipeline_point_count() -> None:
    result = run_pipeline(PipelineConfig(grid_size=20), verbose=False)
    assert len(result.grid_points) == 20 * 20
    assert len(result.displaced_points) == 20 * 20
    assert len(result.polyline) == 20 * 20
    assert result.line_mesh.n_points == 20 * 20


def test_zero_amplitudes_keep_flat_plane() -> None:
    cfg = PipelineConfig(
        grid_size=10,
        amplitude_sw=0.0,
        amplitude_se=0.0,
        amplitude_ne=0.0,
        amplitude_nw=0.0,
    )
    result = run_pipeline(cfg, verbose=False)
    assert np.allclose(result.displacement, 0.0)
    assert np.allclose(result.displaced_points[:, 2], 0.0)


def test_amplitude_changes_field() -> None:
    base = run_pipeline(PipelineConfig(grid_size=16, amplitude_sw=1.0), verbose=False)
    boosted = run_pipeline(PipelineConfig(grid_size=16, amplitude_sw=2.0), verbose=False)
    assert not np.allclose(base.displacement, boosted.displacement)


def test_serpentine_connectivity_starts_at_sw() -> None:
    result = run_pipeline(PipelineConfig(grid_size=5, side_length=10.0), verbose=False)
    assert np.allclose(result.polyline[0], [0.0, 0.0, result.polyline[0, 2]])

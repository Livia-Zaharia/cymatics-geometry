"""cymatics-geometry: plane-of-points displaced by multi-source wave interference."""

from cymatics_geometry.config import (
    PipelineConfig,
    list_saved_configs,
    load_pipeline_config,
    load_voxel_params,
    save_model_params,
    save_pipeline_config,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SectionBox",
    "VoxelPipeConfig",
    "VoxelPipeResult",
    "export_line_obj",
    "export_line_ply",
    "export_stl",
    "list_saved_configs",
    "load_and_place_shape",
    "load_pipeline_config",
    "load_shape_2d",
    "load_voxel_params",
    "pipe_and_export_stl",
    "pipe_lines_to_voxels",
    "run_pipeline",
    "save_model_params",
    "save_pipeline_config",
]


def __getattr__(name: str):
    """Lazy exports so `import cymatics_geometry.config` does not pull pipeline/grid."""
    if name in {"PipelineResult", "export_line_obj", "export_line_ply", "run_pipeline"}:
        from cymatics_geometry import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in {
        "VoxelPipeConfig",
        "VoxelPipeResult",
        "export_stl",
        "pipe_and_export_stl",
        "pipe_lines_to_voxels",
    }:
        from cymatics_geometry import voxels as _voxels

        return getattr(_voxels, name)
    if name in {"SectionBox"}:
        from cymatics_geometry.crop import SectionBox

        return SectionBox
    if name in {"load_shape_2d", "load_and_place_shape"}:
        from cymatics_geometry import custom_shape as _custom

        return getattr(_custom, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

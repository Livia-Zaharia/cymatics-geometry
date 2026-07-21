"""cymatics-geometry: plane-of-points displaced by multi-source wave interference."""

from cymatics_geometry.config import (
    PipelineConfig,
    list_saved_configs,
    load_pipeline_config,
    save_pipeline_config,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "export_line_obj",
    "export_line_ply",
    "list_saved_configs",
    "load_pipeline_config",
    "run_pipeline",
    "save_pipeline_config",
]


def __getattr__(name: str):
    """Lazy exports so `import cymatics_geometry.config` does not pull pipeline/grid."""
    if name in {"PipelineResult", "export_line_obj", "export_line_ply", "run_pipeline"}:
        from cymatics_geometry import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

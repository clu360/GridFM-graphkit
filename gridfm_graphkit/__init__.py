"""
Lightweight package init for GridFM GraphKit.

Submodules are imported lazily by the helper functions in
`gridfm_graphkit.io.param_handler` so inference-only workflows do not pull in
training dependencies unless they are actually needed.
"""

__all__ = [
    "gridfm_graphkit",
]

"""
Some useful and simple helpers for hppfcl:
- load_hppfcl_convex_from_stl_file
"""

import hppfcl


def load_hppfcl_convex_from_stl_file(path: str) -> hppfcl.ConvexBase:
    """
    Load a convex hppfcl object from .stl file whose path is given in argument.
    """
    shape: hppfcl.ConvexBase
    loader = hppfcl.MeshLoader()
    mesh_: hppfcl.BVHModelBase = loader.load(path)
    mesh_.buildConvexHull(True, "Qt")
    shape = mesh_.convex
    return shape

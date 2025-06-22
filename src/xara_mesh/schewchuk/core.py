"""
Pure-Python replacement for cytriangle.CyTriangle
"""
from __future__ import annotations
from ctypes import byref
from ._triangle_ct import _triangulate
from .trio import TriangleIO
import re

class CyTriangle:
    def __init__(self, input_dict=None):
        self._in   = TriangleIO(input_dict)
        self._out  = TriangleIO()
        self._vor  = TriangleIO()

    # ------------------------------------------------ validation identical --
    @staticmethod
    def _validate_flags(opts):
        if "r" in opts and "triangles" not in opts:
            pass  # keep original logic if you want

    # ------------------------------------------------ internal dispatcher ---
    def _run(self, extra_flags: str, verbose=False):
        opts = (('V' if verbose else 'Q') + 'z' + extra_flags).encode()
        status = _triangulate(opts,
                              byref(self._in._io),
                              byref(self._out._io),
                              byref(self._vor._io))
        if status != 0:
            raise RuntimeError(f"triangulate() returned {status}")
        self._out.collect_after_call()
        self._vor.collect_after_call()
        return self._out

    # ------------------------------------------------ public API ------------
    def triangulate(self, flags='', verbose=False):
        return self._run(flags, verbose)

    def delaunay(self, verbose=False):
        return self._run('', verbose)

    def convex_hull(self, verbose=False):
        return self._run('c', verbose)

    def voronoi(self, verbose=False):
        return self._run('v', verbose)

# convenience top-level helper like old triangulate()
def triangulate(input_dict, flags):
    tri = CyTriangle(input_dict)
    tri.triangulate(flags)
    return tri._out.to_dict(np_fmt=True)


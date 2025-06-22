"""
Pure-ctypes bridge to Jonathan Shewchuk’s Triangle
"""
from __future__ import annotations
from ctypes import *
from pathlib import Path
import numpy as np



_pkg_dir = Path(__file__).resolve().parent

def _lib_path() -> str:
    if sys.platform.startswith("linux"):
        name = "libtriangle_lib.so"
    elif sys.platform == "darwin":
        name = "libtriangle_lib.dylib"
    else:                         # win32 / msys / mingw
        name = "triangle_lib.dll"
    return str(_pkg_dir / name)

_lib = CDLL(_lib_path())


# _lib = CDLL(str(Path(__file__).with_suffix("").with_name("libtriangle.so")))

REAL = c_double
c_int_p    = POINTER(c_int)
c_real_p   = POINTER(REAL)

class TriangulateIO(Structure):
    _fields_ = [                                           # ↱ header order
        ("pointlist",            c_real_p),                #   |
        ("pointattributelist",   c_real_p),                #   |
        ("pointmarkerlist",      c_int_p ),                #   |
        ("numberofpoints",       c_int  ),                 #   |
        ("numberofpointattributes", c_int),                #   |
        ("trianglelist",         c_int_p),                 #   |
        ("triangleattributelist",c_real_p),                #   |
        ("trianglearealist",     c_real_p),                #   |
        ("neighborlist",         c_int_p),                 #   |
        ("numberoftriangles",    c_int  ),                 #   |
        ("numberofcorners",      c_int  ),                 #   |
        ("numberoftriangleattributes", c_int),             #   |
        ("segmentlist",          c_int_p),                 #   |
        ("segmentmarkerlist",    c_int_p),                 #   |
        ("numberofsegments",     c_int  ),                 #   |
        ("holelist",             c_real_p),                #   |
        ("numberofholes",        c_int  ),                 #   |
        ("regionlist",           c_real_p),                #   |
        ("numberofregions",      c_int  ),                 #   |
        ("edgelist",             c_int_p),                 #   |
        ("edgemarkerlist",       c_int_p),                 #   |
        ("normlist",             c_real_p),                #   |
        ("numberofedges",        c_int  ),                 #
    ]

#
# prototypes
#
_triangulate = _lib.triangulate
_triangulate.argtypes = [c_char_p,
                         POINTER(TriangulateIO),
                         POINTER(TriangulateIO),
                         POINTER(TriangulateIO)]
_triangulate.restype  = c_int      # Triangle returns 0 on success

_trifree = _lib.trifree
_trifree.argtypes = [c_void_p]
_trifree.restype  = None



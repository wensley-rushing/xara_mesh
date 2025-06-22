"""
Pure-Python equivalent of cytriangleio.TriangleIO
================================================

Responsible for:
  * allocating / filling a _triangle_ct.TriangulateIO struct for input
  * owning NumPy buffers so they don’t get GC’ed mid-call
  * freeing Triangle-allocated output with trifree()
  * convenient to_dict() just like the old Cython version
"""
from __future__ import annotations
import numpy as np
from ctypes import c_int, c_double, c_void_p, POINTER, sizeof, byref
from ._triangle_ct import TriangulateIO, _trifree, REAL

c_int_p   = POINTER(c_int)
c_real_p  = POINTER(REAL)

class TriangleIO:
    # --------------------------------------------------------------------- #

    def __init__(self, d: dict | None = None) -> None:
        self._io = TriangulateIO()     # all fields NULL/0 by default
        self._py_buffers: list[np.ndarray] = []   # own numpy → avoid GC
        self._c_to_free:  list[c_void_p]   = []   # Triangle-malloc’d output
        if d:
            self._from_dict(d)

    # ------------------ convenience -------------------------------------- #
    def _keep(self, arr: np.ndarray) -> c_real_p | c_int_p:
        """keep numpy buffer alive and return pointer cast for struct"""
        self._py_buffers.append(arr)
        return arr.ctypes.data_as(c_real_p if arr.dtype == np.float64 else c_int_p)

    # ------------------ Setters used in old .pyx ------------------------- #
    ## vertices ##
    def set_vertices(self, verts):
        verts = np.ascontiguousarray(verts, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("vertices must be (N,2) array-like")
        self._io.numberofpoints = verts.shape[0]
        self._io.pointlist      = self._keep(verts)

    ## optional per-vertex ##
    def set_vertex_attributes(self, attr):
        attr = np.ascontiguousarray(attr, dtype=np.float64)
        if attr.shape[0] != self._io.numberofpoints:
            raise ValueError("vertex_attributes length mismatch")
        self._io.numberofpointattributes = attr.shape[1]
        self._io.pointattributelist      = self._keep(attr)

    def set_vertex_markers(self, mk):
        mk = np.ascontiguousarray(mk, dtype=np.int32)
        if mk.size != self._io.numberofpoints:
            raise ValueError("vertex_markers length mismatch")
        self._io.pointmarkerlist = self._keep(mk)

    ## triangles + per-triangle ##
    def set_triangles(self, tris):
        tris = np.ascontiguousarray(tris, dtype=np.int32)
        self._io.numberoftriangles = tris.shape[0]
        self._io.numberofcorners   = tris.shape[1]
        self._io.trianglelist      = self._keep(tris)

    def set_triangle_attributes(self, attr):
        attr = np.ascontiguousarray(attr, dtype=np.float64)
        if attr.shape[0] != self._io.numberoftriangles:
            raise ValueError("triangle_attributes length mismatch")
        self._io.numberoftriangleattributes = attr.shape[1]
        self._io.triangleattributelist      = self._keep(attr)

    def set_triangle_areas(self, a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        if a.size != self._io.numberoftriangles:
            raise ValueError("triangle_max_area length mismatch")
        self._io.trianglearealist = self._keep(a)

    ## segments / holes / regions (similar but shorter checks) ##
    def set_segments(self, seg):
        seg = np.ascontiguousarray(seg, dtype=np.int32)
        if seg.shape[1] != 2:
            raise ValueError("segments must be (N,2)")
        self._io.numberofsegments = seg.shape[0]
        self._io.segmentlist      = self._keep(seg)

    def set_segment_markers(self, mk):
        mk = np.ascontiguousarray(mk, dtype=np.int32)
        if mk.size != self._io.numberofsegments:
            raise ValueError("segment_markers length mismatch")
        self._io.segmentmarkerlist = self._keep(mk)

    def set_holes(self, holes):
        holes = np.ascontiguousarray(holes, dtype=np.float64)
        if holes.shape[1] != 2:
            raise ValueError("holes must be (N,2)")
        self._io.numberofholes = holes.shape[0]
        self._io.holelist      = self._keep(holes)

    def set_regions(self, regs):
        # expect list[dict] as in old wrapper
        arr = np.empty((len(regs), 4), np.float64)
        for i,r in enumerate(regs):
            arr[i] = [r["vertex"][0], r["vertex"][1], r["marker"], r["max_area"]]
        self._io.numberofregions = arr.shape[0]
        self._io.regionlist      = self._keep(arr)

    # ------------------ dict ↔︎ struct helpers --------------------------- #
    def _from_dict(self, d):
        if "vertices" in d:            self.set_vertices(d["vertices"])
        if "vertex_attributes" in d:   self.set_vertex_attributes(d["vertex_attributes"])
        if "vertex_markers" in d:      self.set_vertex_markers(d["vertex_markers"])
        if "triangles" in d:
            self.set_triangles(d["triangles"])
            if "triangle_attributes" in d:
                self.set_triangle_attributes(d["triangle_attributes"])
            if "triangle_max_area" in d:
                self.set_triangle_areas(d["triangle_max_area"])
        if "segments" in d:
            self.set_segments(d["segments"])
            if "segment_markers" in d:
                self.set_segment_markers(d["segment_markers"])
        if "holes" in d:               self.set_holes(d["holes"])
        if "regions" in d:             self.set_regions(d["regions"])

    # ---- collect output after triangulate() ----------------------------- #
    def to_dict(self, np_fmt: bool = False):
        out = {}
        if self._io.pointlist:
            pts = np.ctypeslib.as_array(
                self._io.pointlist,
                shape=(self._io.numberofpoints, 2))
            out["vertices"] = pts.copy() if np_fmt else pts.tolist()
        if self._io.trianglelist:
            tris = np.ctypeslib.as_array(
                self._io.trianglelist,
                shape=(self._io.numberoftriangles,
                       self._io.numberofcorners))
            out["triangles"] = tris.copy() if np_fmt else tris.tolist()
        # add other arrays exactly the way you need them
        return out

    # ------------------ resource cleanup --------------------------------- #
    def _take_cptr(self, ptr):
        """record Triangle-malloc’d pointer so we can free later"""
        if ptr: self._c_to_free.append(ptr)

    def collect_after_call(self):
        """call *once* in the _out / _vor instances after triangulate()
        to remember which arrays Triangle allocated"""
        # Triangulate may allocate lots of arrays; record pointers that
        # are currently non-NULL but **not** owned by Python.
        io = self._io
        for name in ("pointlist","pointattributelist","pointmarkerlist",
                     "trianglelist","triangleattributelist","trianglearealist",
                     "neighborlist","segmentlist","segmentmarkerlist",
                     "holelist","regionlist","edgelist","edgemarkerlist",
                     "normlist"):
            ptr = getattr(io, name)
            if ptr and not any(ptr == buf.ctypes.data for buf in self._py_buffers):
                self._take_cptr(ptr)

    def __del__(self):
        for p in self._c_to_free:
            _trifree(p)


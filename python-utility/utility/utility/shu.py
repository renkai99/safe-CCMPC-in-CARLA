"""Shapely operations"""
import shapely
import shapely.geometry

def vertex_set_to_smpoly(vertex_set):
    """Sets of vertices to Shapely MultiPolygon."""
    polygons = []
    for vertices in vertex_set:
        polygons.append([vertices, []])
    return shapely.geometry.MultiPolygon(polygons)


def vertices_to_smpoly(vertices):
    """Vertices to Shapely MultiPolygon"""
    polygons = [[vertices, []]]
    return shapely.geometry.MultiPolygon(polygons)

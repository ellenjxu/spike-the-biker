from typing import List

import ezdxf
import numpy as np
import shapely.geometry as geom


def lwpolyline2xy(polyline, degrees_per_segment):
    xy = []

    points = polyline.get_points()

    for i, point in enumerate(points):
        x, y, _, _, b = point
        xy.append([x, y])

        if b != 0:  # if bulge
            # if next point is the end, next point is the start bc closed
            if i + 1 == len(points):
                next_point = points[0]
            else:
                next_point = points[i + 1]

            p1 = [x, y]
            p2 = [next_point[0], next_point[1]]

            pts = arc_points_from_bulge(p1, p2, b, degrees_per_segment)

            # exclude start and end points
            # start point was already added above
            # last point is next point, will be added when dealing with the next point
            pts = pts[1:-1]

            xy.extend(pts)
    return xy


def lwpolyline2ring(
    polyline: ezdxf.entities.LWPolyline, degrees_per_segment: float = 0.5
) -> geom.LinearRing:
    """lwpolyline is a lightweight polyline (cf POLYLINE) 
    modified from: https://github.com/aegis1980/cad-to-
    shapely/blob/master/cad_to_shapely/dxf.py."""

    return geom.LinearRing(lwpolyline2xy(polyline, degrees_per_segment))


def lwpolyline2string(
    polyline: ezdxf.entities.LWPolyline, degrees_per_segment: float = 0.5
) -> geom.LinearRing:
    """lwpolyline is a lightweight polyline (cf POLYLINE) 
    modified from: https://github.com/aegis1980/cad-to-
    shapely/blob/master/cad_to_shapely/dxf.py."""

    return geom.LineString(lwpolyline2xy(polyline, degrees_per_segment))


def arc_points(
    start_angle: float,
    end_angle: float,
    radius: float,
    center: List[float],
    degrees_per_segment: float,
) -> list:
    """Coordinates of an arcs (for approximation as a polyline)

    Args:
        start_angle (float): arc start point relative to centre, in radians
        end_angle (float): arc end point relative to centre, in radians
        radius (float): [description]
        center (List[float]): arc centre as [x,y]
        degrees_per_segment (float): [description]

    Returns:
        list: 2D list of points as [x,y]

    from https://github.com/aegis1980/cad-to-shapely/blob/master/cad_to_shapely/utils.py
    """

    n = abs(int((end_angle - start_angle) / np.radians(degrees_per_segment)))  # number of segments
    theta = np.linspace(start_angle, end_angle, n)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return np.column_stack([x, y])


def distance(p1: List[float], p2: List[float]) -> float:
    """from https://github.com/aegis1980/cad-to-shapely/blob/master/cad_to_shapely/utils.py."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def arc_points_from_bulge(p1: List[float], p2: List[float], b: float, degrees_per_segment: float):
    """http://darrenirvine.blogspot.com/2015/08/polylines-radius-bulge-turnaround.html.

    Args:
        p1 (List[float]): [description]
        p2 (List[float]): [description]
        b (float): bulge of the arc
        degrees_per_segment (float): [description]

    Returns:
        [type]: point on arc

    from: https://github.com/aegis1980/cad-to-shapely/blob/master/cad_to_shapely/utils.py
    """

    theta = 4 * np.arctan(b)
    u = distance(p1, p2)

    r = u * ((b**2) + 1) / (4 * b)

    try:
        a = np.sqrt(r**2 - (u * u / 4))
    except ValueError:
        a = 0

    dx = (p2[0] - p1[0]) / u
    dy = (p2[1] - p1[1]) / u

    A = np.array(p1)
    B = np.array(p2)
    # normal direction
    N = np.array([dy, -dx])

    # if bulge is negative arc is clockwise
    # otherwise counter-clockwise
    s = b / abs(b)  # sigma = signum(b)

    # centre, as a np.array 2d point

    if abs(theta) <= np.pi:
        C = ((A + B) / 2) - s * a * N
    else:
        C = ((A + B) / 2) + s * a * N

    start_angle = np.arctan2(p1[1] - C[1], p1[0] - C[0])
    if b < 0:
        start_angle += np.pi

    end_angle = start_angle + theta

    return arc_points(start_angle, end_angle, r, C, degrees_per_segment)
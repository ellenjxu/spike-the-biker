from trajectory_generation.utils import lwpolyline2ring, lwpolyline2string
import shapely.geometry as geom
import ezdxf

class Floorplan:
    def __init__(self, floorplan_fpath, centerline_fpath):
        self.floor_outer = lwpolyline2ring(self._get_polyline_from_dxf(floorplan_fpath))
        self.floor = geom.Polygon(shell=self.floor_outer)

        centerline = self._get_polyline_from_dxf(centerline_fpath)
        self.centerline = lwpolyline2ring(centerline)

        self.centerline_polygon = geom.Polygon(shell=self.centerline)

    def _get_polyline_from_dxf(self, fpath):
        """
        load a single polyline from dxf, assumes there is at least one
        """
        doc = ezdxf.readfile(fpath)
        msp = list(doc.modelspace())
        entity = msp[0]
        return entity

    def is_inside(self, x, y):
        """given x, y coords, check if they are inside the track (between inner and outer)"""
        point = geom.Point(x, y)
        return self.floor.contains(point)

    def distance_to_centerline(self, x, y):
        point = geom.Point(x, y)
        return self.centerline.distance(point)

    def get_progress(self, x, y):
        """project x,y on centerline and get track progress from the start of the centerline."""
        point = geom.Point(x, y)
        return self.centerline.project(point)
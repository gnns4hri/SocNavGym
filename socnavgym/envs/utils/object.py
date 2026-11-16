from abc import abstractmethod
import math
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, Point
from shapely import affinity
from socnavgym.envs.utils.utils import get_coordinates_of_rotated_line, get_coordinates_of_rotated_rectangle

_ROUND = ("plant", "robot", "human")
_RECT = ("laptop", "table", "chair", "wall")


def _surely_apart(a, b, eps=1e-6):
    """True only when the shapes used by Object.collides cannot intersect, so that the (slow) shapely test can be skipped
    without changing any result. Conservative: every shape lies inside the bound used here (the polygon that shapely builds
    for a circle is inscribed in it; a human's ellipse lies inside the circle of radius width/2), and anything within `eps`
    of touching is left to shapely. Added because rejection sampling of goals (sample_goal) calls collides thousands of
    times per step when the map is crowded."""
    an, bn = a.name, b.name
    try:
        if an in _ROUND:
            ra = a.radius if an != "human" else a.width / 2
            if bn in _ROUND: return math.hypot(a.x - b.x, a.y - b.y) > ra + (b.radius if bn != "human" else b.width / 2) + eps
            if bn in _RECT: return _far_from_rect(a.x, a.y, ra + eps, b)
        elif an in _RECT and bn in _ROUND:
            return _far_from_rect(b.x, b.y, (b.radius if bn != "human" else b.width / 2) + eps, a)
        elif an in _RECT and bn in _RECT:
            ha = 0.5 * math.hypot(a.length, a.thickness if an == "wall" else a.width)
            hb = 0.5 * math.hypot(b.length, b.thickness if bn == "wall" else b.width)
            return math.hypot(a.x - b.x, a.y - b.y) > ha + hb + eps
    except (TypeError, AttributeError): pass               # unset attributes: let the original code raise its own errors
    return False


_INNER = 0.99   # the polygon shapely builds for a circle of radius r contains the circle of radius r*cos(pi/32) = 0.9952 r (8 segments
                # per quarter, the coarsest default of any shapely version; finer ones contain more)


def _surely_overlapping(a, b, eps=1e-6):
    """True only when the shapes used by Object.collides certainly intersect (the counterpart of _surely_apart): each round
    shape contains a circle around its centre (radius _INNER * r; for a human's ellipse, _INNER * width/6, its short
    semi-axis); if those inner circles overlap each other, or reach a rectangle, the shapes intersect."""
    an, bn = a.name, b.name
    try:
        if an in _ROUND:
            ra = _INNER * (a.radius if an != "human" else a.width / 6)
            if bn in _ROUND: return math.hypot(a.x - b.x, a.y - b.y) < ra + _INNER * (b.radius if bn != "human" else b.width / 6) - eps
            if bn in _RECT: return not _far_from_rect(a.x, a.y, ra - eps, b)
        elif an in _RECT and bn in _ROUND:
            return not _far_from_rect(b.x, b.y, _INNER * (b.radius if bn != "human" else b.width / 6) - eps, a)
    except (TypeError, AttributeError): pass
    return False


def _far_from_rect(px, py, r, rect):
    """Is the point (px, py) farther than r from the rectangle (centre x, y; `length` along its orientation)?"""
    c, s = math.cos(rect.orientation), math.sin(rect.orientation); dx, dy = px - rect.x, py - rect.y
    qx = abs(dx * c + dy * s) - rect.length / 2
    qy = abs(-dx * s + dy * c) - (rect.thickness if rect.name == "wall" else rect.width) / 2
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) > r


class Object(object):
    """
    Base class for the objects in the environment.

    Objects :
    - tables
    - laptops
    - plants
    - walls
    - humans
    - robot

    """

    def __init__(self, id:int, name: str) -> None:
        self.id = id  # unique identifier for the object
        self.x = None  # x coordinate (generally the x-coordinate of the center of mass)
        self.y = None  # y coordinate (generally the y-coordinate of the center of mass)
        self.orientation = None  # angle with the X-axis of the global frame
        self.name = name  # string representing the type of object
        self.is_static = (
            True  # boolean variable denoting whether the object is static or dynamic
        )
        # color of the object for drawing purposes
        self.color = None
        # mass of the object
        self.mass = 1.0
        # one-hot encoding for the object.
        self.one_hot_encoding = None

        if self.name == "robot":
            self.one_hot_encoding = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)

        elif self.name == "human":
            self.one_hot_encoding = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)

        elif self.name == "table":
            self.one_hot_encoding = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)

        elif self.name == "laptop":
            self.one_hot_encoding = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)

        elif self.name == "plant":
            self.one_hot_encoding = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)

        elif self.name == "wall":
            self.one_hot_encoding = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)

    def set(self, id, x, y, theta):
        """
        Method for setting attributes of the object
        """
        self.id = id
        self.x = x
        self.y = y
        self.orientation = theta

    def set_color(self, color):
        """
        Method to set the color of the object
        """
        self.color = color

    def get_position(self):
        """
        To get the coordinates of the object
        """
        return (self.x, self.y)

    def get_orientation(self):
        """
        To get the orientation of the object
        """
        return self.orientation

    def get_name(self):
        """
        To get the name (type) of the object
        """
        return self.name

    def collides(self, obj):
        """
        A boolean function that determines if there is a collision between current object and the given object
        """

        if _surely_apart(self, obj): return False   # exact shortcut: same answer as the shapely test below, far cheaper
        if _surely_overlapping(self, obj): return True

        if self.name == "plant" or self.name == "robot":
            assert(self.x != None and self.y != None and self.radius != None), "Attributes are None type"
            curr_obj = Point((self.x, self.y)).buffer(self.radius)

        elif self.name == "human":
            assert(self.x != None and self.y != None and self.width != None), "Attributes are None type"
            curr_obj = shapely.affinity.rotate(shapely.affinity.scale(shapely.geometry.Point((self.x, self.y)).buffer(1), self.width/6, self.width/2), self.orientation, origin=(self.x, self.y), use_radians=True)  # ellipse: front-back self.width/6, shoulders self.width/2, rotated with the heading

        elif self.name == "laptop" or self.name == "table" or self.name == "chair":
            assert(self.x != None and self.y != None and self.width != None and self.length != None and self.orientation != None), "Attributes are None type"
            curr_obj = Polygon(get_coordinates_of_rotated_rectangle(self.x, self.y, self.orientation, self.length, self.width))

        elif self.name == "wall":
            assert(self.x != None and self.y != None and self.thickness != None and self.length != None and self.orientation != None), "Attributes are None type"
            curr_obj = Polygon(get_coordinates_of_rotated_rectangle(self.x, self.y, self.orientation, self.length, self.thickness))

        else: raise NotImplementedError


        if obj.name == "plant" or obj.name == "robot":
            assert(obj.x != None and obj.y != None and obj.radius != None), "Attributes are None type"
            other_obj = Point((obj.x, obj.y)).buffer(obj.radius)

        elif obj.name == "human":
            assert(obj.x != None and obj.y != None and obj.width != None), "Attributes are None type"
            other_obj = shapely.affinity.rotate(shapely.affinity.scale(shapely.geometry.Point((obj.x, obj.y)).buffer(1), obj.width/6, obj.width/2), obj.orientation, origin=(obj.x, obj.y), use_radians=True)  # ellipse: front-back obj.width/6, shoulders obj.width/2, rotated with the heading

        elif obj.name == "laptop" or obj.name == "table" or obj.name == "chair":
            assert(obj.x != None and obj.y != None and obj.width != None and obj.length != None and obj.orientation != None), "Attributes are None type"
            other_obj = Polygon(get_coordinates_of_rotated_rectangle(obj.x, obj.y, obj.orientation, obj.length, obj.width))

        elif obj.name == "wall":
            assert(obj.x != None and obj.y != None and obj.thickness != None and obj.length != None and obj.orientation != None), "Attributes are None type"
            other_obj = Polygon(get_coordinates_of_rotated_rectangle(obj.x, obj.y, obj.orientation, obj.length, obj.thickness))

        elif obj.name == "human-human-interaction":
            return obj.collides(self)

        elif obj.name == "human-laptop-interaction":
            return obj.collides(self)

        else: raise NotImplementedError

        return curr_obj.intersects(other_obj)

    @abstractmethod
    def draw(self, img, PIXEL_TO_WORLD, MAP_SIZE, color=None, radius=None, nose=None):
        """
        Function for drawing the object on the image.
        """
        return

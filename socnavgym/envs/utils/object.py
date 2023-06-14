from abc import abstractmethod
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely import affinity
from socnavgym.envs.utils.utils import get_coordinates_of_rotated_line, get_coordinates_of_rotated_rectangle

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

        if self.name == "plant" or self.name == "robot":
            assert(self.x != None and self.y != None and self.radius != None), "Attributes are None type"
            curr_obj = Point((self.x, self.y)).buffer(self.radius)
        
        elif self.name == "human":
            assert(self.x != None and self.y != None and self.width != None), "Attributes are None type"
            curr_obj = Point((self.x, self.y)).buffer(self.width/2)
        
        elif self.name == "laptop" or self.name == "table":
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
            other_obj = Point((obj.x, obj.y)).buffer(obj.width/2)
        
        elif obj.name == "laptop" or obj.name == "table":
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

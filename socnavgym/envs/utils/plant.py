import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py


class Plant(Object):
    """
    Class for Plant
    """

    def __init__(self, id=None, x=None, y=None, radius=None) -> None:
        super().__init__(id, "plant")
        self.radius = None # radius of the plant
        self.set(id, x, y, 0, radius)

    def set(self, id, x, y, theta, radius):
        super().set(id, x, y, theta)
        self.radius = radius

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        brown = (29, 67, 105)  # brown
        green = (0, 200, 0) # green
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        ) # calculating the number of pixels corresponding to the radius

        cv2.circle(
            img,
            (
                w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y),
            ),
            radius,
            brown,
            -1,
        ) # drawing a brown circle for the pot in which the plant is kept
        cv2.circle(
            img,
            (
                w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y),
            ),
            int(radius / 2),
            green,
            -1,
        )  # drawing a green circle for the plant

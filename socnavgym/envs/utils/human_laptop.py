import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.human import Human
from socnavgym.envs.utils.laptop import Laptop
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2
from typing import List
import random

class Human_Laptop_Interaction:
    """
    Class for Human-Laptop interaction
    """

    def __init__(self, laptop:Laptop, distance, human_width, can_disperse=True) -> None:
        self.name = "human-laptop-interaction"
        # laptop
        self.laptop = laptop

        self.x = None
        self.y = None
       
        self.can_disperse = can_disperse

        # generating a human 
        self.human = Human(speed=0, width=human_width,policy=random.choice(["orca", "sfm"]))
        
        # distance between the human and laptop centers
        self.distance = distance
        
        # arranging the human
        self.arrange_human()

    def arrange_human(self):
        self.human.x = self.laptop.x + np.cos(self.laptop.orientation - np.pi/2) * self.distance
        self.human.y = self.laptop.y + np.sin(self.laptop.orientation - np.pi/2) * self.distance
        self.human.orientation = self.laptop.orientation + np.pi/2
        self.x = (self.laptop.x + self.human.x)/2
        self.y = (self.laptop.y + self.human.y)/2


    def collides(self, obj, human_only=False):
        if obj.name == "human-human-interaction":
            for h in obj.humans:
                if self.human.collides(h):
                    return True
            return False
        
        elif obj.name == "human-laptop-interaction":
            if self.human.collides(obj.human): 
                return True
            
            elif self.laptop.collides(obj.laptop):
                return True
            
            return False
        
        else:
            return self.human.collides(obj) or (self.laptop.collides(obj) and (not human_only))

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        self.laptop.draw(img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y)
        self.human.draw(img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y)

        points = []
        points.append(
            [
                w2px(self.laptop.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                w2py(self.laptop.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
            ]
        )

        points.append(
            [
                w2px(self.human.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                w2py(self.human.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
            ]
        )
        points = np.array(points).reshape((-1,1,2))
        cv2.polylines(img, [np.int32(points)], True, (0, 0, 255), 1)

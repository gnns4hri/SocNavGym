import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py


class Chair(Object):
    """
    Class for chairs
    """

    def __init__(self, id=None, x=None, y=None, theta=None, width=None, length=None) -> None:
        super().__init__(id, "chair")
        self.width = None  # width of the table
        self.length = None  # length of the table
        self.set(id, x, y, theta, width, length)

    def set(self, id, x, y, theta, width, length):
        super().set(id, x, y, theta)
        self.width = width
        self.length = length

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        if self.color == None:
            color = (100, 100, 100)  # dark gray
        else: color = self.color

        assert (
            self.length != None and self.width != None
        ), "Length or breadth is None type."
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"

        # p1, p2, p3, p4 are the coordinates of the corners of the rectangle. calculation is done so as to orient the rectangle at an angle.

        object_points = []
        part_colors = []

        s1 = 0.1
        s2 = 0.2

        p1 = [-self.width / 2, -self.length / 2]
        p2 = [self.width / 2, -self.length / 2]
        p3 = [self.width / 2, -self.length / 2  + self.length * s1]        
        p4 = [-self.width / 2, -self.length / 2 + self.length * s1]

        part_points = [p1, p2, p3, p4]

        object_points.append(part_points)
        part_colors.append(color)

        p5 = [-self.width / 2, -self.length / 2 + self.length * s2]
        p6 = [-self.width / 2 + self.width * s1 , -self.length / 2 + self.length * s2]
        p7 = [-self.width / 2 + self.width * s1 , self.length / 2]
        p8 = [-self.width / 2, self.length / 2]

        part_points = [p5, p6, p7, p8]

        object_points.append(part_points)
        part_colors.append(color)

        p9 = [self.width / 2, -self.length / 2 + self.length * s2]
        p10 = [self.width / 2 - self.width * s1 , -self.length / 2 + self.length * s2]
        p11 = [self.width / 2 - self.width * s1 , self.length / 2]
        p12 = [self.width / 2, self.length / 2]

        part_points = [p9, p10, p11, p12]

        object_points.append(part_points)
        part_colors.append(color)

        p13 = [-self.width / 2 + self.width * s2, -self.length / 2 + self.length * s2]
        p14 = [self.width / 2 - self.width * s2, -self.length / 2 + self.length * s2]
        p15 = [self.width / 2 - self.width * s2, self.length / 2]
        p16 = [-self.width / 2 + self.width * s2, self.length / 2]

        part_points = [p13, p14, p15, p16]

        object_points.append(part_points)
        part_colors.append((200, 200, 200))

        im_op = []
        for op in object_points:
            pp = []
            for p in op:
                ip = [0]*2
                ip[0] =  w2px((self.x - p[0] * np.sin(self.orientation) + p[1] * np.cos(self.orientation)),PIXEL_TO_WORLD_X, MAP_SIZE_X)
                ip[1] =  w2py((self.y + p[0] * np.cos(self.orientation) + p[1] * np.sin(self.orientation)),PIXEL_TO_WORLD_X, MAP_SIZE_X)
                pp.append(ip)
            im_op.append(pp)

        for ip, c in zip(im_op, part_colors):
            points = np.array(ip)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(img, [np.int32(points)], c, cv2.LINE_AA)  # filling the rectangle made from the points with the specified color
            cv2.polylines(img, [np.int32(points)], True, (0, 0, 0), 2, cv2.LINE_AA)  # bordering the rectangle

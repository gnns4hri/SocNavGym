import sys, os
import json
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import QRect
import subprocess
import math
import numpy as np

import time

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from socnav import *


import ui_drawgraph

def node_type(typemap):
    mapping = {
        'g': 'grid',
        'p': 'human',
        'o': 'object',
        'w': 'wall',
        'l': 'room',
        't': 'target'
    }
    return mapping[typemap]


class MainClass(QtWidgets.QWidget):
    def __init__(self, scenarios, start, alt):
        super().__init__()
        self.scenarios = scenarios
        self.alt = alt
        all_features, n_features = get_features()
        self.ui = ui_drawgraph.Ui_SocNavWidget()
        self.ui.setupUi(self)
        self.next_index = start
        self.view = None
        self.show()
        self.installEventFilter(self)
        self.load_next()
        self.ui.tableWidget.setRowCount(self.view.graph.ndata['h'].shape[1]+1)
        self.ui.tableWidget.setColumnCount(1)
        self.ui.tableWidget.setColumnWidth(0, 200)
        self.ui.tableWidget.show()

        # Initialize table
        self.ui.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('value'))
        self.ui.tableWidget.horizontalHeader().hide()
        self.ui.tableWidget.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('type'))
        self.ui.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem('0'))
        features_aux = self.view.graph.ndata['h'][1]
        for idx, feature in enumerate(features_aux, 1):
            self.ui.tableWidget.setVerticalHeaderItem(idx, QtWidgets.QTableWidgetItem(all_features[idx-1]))
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

    def next(self, next_scenarios):
        self.scenarios = next_scenarios
        self.next_index = 0
        self.load_next()

    def load_next(self):
        if self.next_index >= len(self.scenarios):
            print("All graphs shown")
            sys.exit(0)
        typemap = self.scenarios.data['typemaps'][self.next_index]
        coordinates = self.scenarios.data['coordinates'][self.next_index]
        n_frames = 1
        view = MyView(self.scenarios[self.next_index], typemap, coordinates, n_frames,
                           self.ui.tableWidget, self.alt)
        if hasattr(self.view, 'view_closest_node_id'):
            view.view_closest_node_id = self.view.view_closest_node_id
            view.view_n_type = self.view.view_n_type
        if hasattr(self, 'view'):
            if self.view:
                self.view.close()
            del self.view
        self.view = view
        self.next_index += 1
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())
        self.show()

        # Initialize table with zeros
        for idx in range(self.view.graph.ndata['h'].shape[1]+2):
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))
        if hasattr(self.view, 'view_closest_node_id'):
            self.view.select_node_callback()

        # Uncomment to show the label
        # img = self.scenarios[self.next_index-1][1].cpu().detach().numpy()*255
        # plt.imshow(cv2.cvtColor(img.reshape((150,150)).astype(np.ubyte), cv2.COLOR_GRAY2RGB))
        # plt.show()


    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                sys.exit(0)
            else:
                if key == QtCore.Qt.Key_Return:
                    self.load_next()
                elif key == QtCore.Qt.Key_Enter:
                    self.close()
                return True
        return False


class MyView(QtWidgets.QGraphicsView):
    def __init__(self, scenario, typemap, coordinates, n_frames, table, alt):
        super().__init__()
        self.alt = alt
        self.all_features, self.n_features = get_features()
        self.table = table
        self.graph = scenario[0]
        self.label = scenario[1]
        self.typemap = typemap
        self.coordinates = coordinates
        self.n_frames = n_frames
        self.scene = QtWidgets.QGraphicsScene(self)
        self.nodeItems = dict()
        self.setFixedSize(1002, 1002)
        self.create_scene()
        self.installEventFilter(self)

    def create_scene(self):

        self.scene.setSceneRect(QtCore.QRectF(-500, -500, 1000, 1000))

        # Draw nodes and print labels
        time_f = 0
        grid_end = False
        grid_n = 0

        for index, n_type in self.typemap.items():
            # p = person
            # r = room
            # o = object
            # w = wall
            # g = grid
            # t = target
            angle = None
            if n_type == 'p':
                colour = QtCore.Qt.blue
                node_radius = 15
                x = self.graph.ndata['h'][index][self.all_features.index('hum_x_pos')] * 200
                y = -self.graph.ndata['h'][index][self.all_features.index('hum_y_pos')] * 200
                angle = math.atan2(self.graph.ndata['h'][index][self.all_features.index('hum_orientation_cos')],
                                   self.graph.ndata['h'][index][self.all_features.index('hum_orientation_sin')])

            elif n_type == 'o':
                colour = QtCore.Qt.green
                node_radius = 10
                x = self.graph.ndata['h'][index][self.all_features.index('obj_x_pos')] * 200
                y = -self.graph.ndata['h'][index][self.all_features.index('obj_y_pos')] * 200
                angle = math.atan2(self.graph.ndata['h'][index][self.all_features.index('obj_orientation_cos')],
                                   self.graph.ndata['h'][index][self.all_features.index('obj_orientation_sin')])
            elif n_type == 'r':
                colour = QtCore.Qt.black
                node_radius = 7
                x = 0
                y = 0
            elif n_type == 'w':
                colour = QtCore.Qt.cyan
                node_radius = 10
                x = self.graph.ndata['h'][index][self.all_features.index('wall_x_pos')] * 200
                y = -self.graph.ndata['h'][index][self.all_features.index('wall_y_pos')] * 200
                angle = math.atan2(self.graph.ndata['h'][index][self.all_features.index('wall_orientation_cos')],
                                   self.graph.ndata['h'][index][self.all_features.index('wall_orientation_sin')])
            elif n_type == 'g':
                colour = QtCore.Qt.darkRed
                node_radius = 10
                x = self.graph.ndata['h'][index][self.all_features.index('goal_x_pos')] * 200
                y = -self.graph.ndata['h'][index][self.all_features.index('goal_y_pos')] * 200
            # elif n_type == 'g':
            #     colour = QtCore.Qt.lightGray
            #     node_radius = 10
            #     x = self.graph.ndata['h'][index][self.all_features.index('grid_x_pos')] * 625
            #     y = -self.graph.ndata['h'][index][self.all_features.index('grid_y_pos')] * 625
            else:
                colour = QtCore.Qt.white #None
                node_radius = 5 #None
                x = None
                y = None

            if x is not None:
                # if n_type != 'g' and grid_end is False:
                #     grid_end = True
                #     grid_n = index
                #     time_f = 0
                # if grid_end and (index-grid_n) % ((self.graph.number_of_nodes()-grid_n) / self.n_frames) == 0:
                #     time_f += 1
                #
                # shift = time_f * (node_radius + 20)
                # x += shift
                item = self.scene.addEllipse(x - node_radius, y - node_radius, node_radius*2,
                                             node_radius*2, brush=colour)
                if angle is not None:
                    pen = QtGui.QPen()
                    pen.setWidth(5)
                    pen.setColor(QtCore.Qt.darkRed)
                    x_dest = x + node_radius * 2 * math.sin(angle)
                    y_dest = y + node_radius * 2 * math.cos(angle)
                    self.scene.addLine(x, y, x_dest, y_dest, pen=pen)
            else:
                print(n_type)
                print("Invalid node")
                sys.exit(0)

            c = (x, y)
            self.nodeItems[index] = (item, c, n_type)

            # Print labels of the nodes
            text = self.scene.addText(n_type)
            text.setDefaultTextColor(QtCore.Qt.magenta)
            text.setPos(*c)

        self.setScene(self.scene)

        # Draw edges
        edges = self.graph.edges()

        for e_id in range(len(edges[0])):
            edge_a = edges[0][e_id].item()
            edge_b = edges[1][e_id].item()

            type_a = self.nodeItems[edge_a][2]
            type_b = self.nodeItems[edge_b][2]

            if edge_a == edge_b:  # No self edges printed
                continue

            ax, ay = self.nodeItems[edge_a][1]
            bx, by = self.nodeItems[edge_b][1]
            pen = QtGui.QPen()
            rel_type = self.graph.edata['rel_type'][e_id].item()
            colour, width = self.type_to_colour_width(rel_type, type_a, type_b)

            if colour is None or width is None:
                print("Error for link between these two types:")
                print(type_a)
                print(type_b)
                sys.exit(0)

            if type_a != 'r' and type_b != 'r':
                pen.setColor(colour)
                pen.setWidth(width)
                self.scene.addLine(ax, ay, bx, by, pen=pen)

      

    @staticmethod
    def type_to_colour_width(rel_type, type1, type2):
        if type1 == type2:
            if type1 == 'g' and type2 == 'g':
                colour = QtCore.Qt.lightGray
                width = 1
            else:
                colour = QtCore.Qt.black
                width = 5
        elif (type1 == 'p' and type2 == 'o') or (type1 == 'o' and type2 == 'p'):
            colour = QtCore.Qt.black
            width = 5
        elif type1 == 'l' or type2 == 'l':
            colour = QtCore.Qt.lightGray
            width = 1
        elif (type1 == 'g' and type2 != 'g') or (type2 == 'g' and type1 != 'g'):
            colour = QtCore.Qt.red
            width = 1
        else:
            colour = QtCore.Qt.red #None
            width = 1 #None

        return colour, width

    def closest_node_view(self, event_x, event_y):
        WINDOW_SIZE = 496
        closest_node = -1
        closest_node_type = -1
        x_mouse = (event_x - WINDOW_SIZE)
        y_mouse = (event_y - WINDOW_SIZE)
        old_dist = WINDOW_SIZE * 2

        for idx, node in self.nodeItems.items():
            x = node[1][0]
            y = node[1][1]
            dist = abs(x - x_mouse) + abs(y - y_mouse)
            if dist < old_dist:
                old_dist = dist
                closest_node = idx
                closest_node_type = node[2]

        return closest_node, closest_node_type

    # def set_pixmap(self, pixmap_path):
    #     pixmap = QtGui.QPixmap(pixmap_path)
    #     pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
    #     pixmap_item.setPos(-455, -500)
    #     pixmap_item.setScale(1.)
    #     self.scene.addItem(pixmap_item)
    #     #  self.scene.addRect(-30, -30, 60, 60, pen=QtGui.QPen(QtCore.Qt.white), brush=QtGui.QBrush(QtCore.Qt.white))

    def select_node_callback(self):
        if not hasattr(self, 'view_closest_node_id'):
            print('We don\'t have it yet')
            return

        closest_node_id = self.view_closest_node_id
        n_type = self.view_n_type

        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(n_type))
        features = self.graph.ndata['h'][closest_node_id]

        # one_hot_nodes = [str(int(x)) for x in features[0:5]]
        # one_hot_times = [str(int(x)) for x in features[5:8]]
        # rest = ['{:1.3f}'.format(x) for x in features[8:len(features)]]
        # features_format = one_hot_nodes + one_hot_times + rest
        features_format = ['{:1.3f}'.format(x) for x in features]

        for idx, feature in enumerate(features_format):
            self.table.setItem(idx, 1, QtWidgets.QTableWidgetItem(feature))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            closest_node_id, n_type = self.closest_node_view(event.x()-7, event.y()-7)
            print(f'Setting node {closest_node_id}, type {n_type}')
            if n_type == -1:
                print('Not valid label')

            self.view_closest_node_id = closest_node_id
            self.view_n_type = n_type
            self.select_node_callback()

            return True
        return False


if __name__ == '__main__':
    alt = '1'

    if sys.argv[1].endswith('.json'):
        with open(sys.argv[1]) as json_file:
            data = json.load(json_file)

        data.reverse()
    else:
        data = sys.argv[1]

    scenarios = SocNavDataset(data, mode='test', raw_dir='', alt=alt, debug=True)

    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    else:
        start = 0

    view = MainClass(scenarios, start, alt=alt)

    exit_code = app.exec_()
    sys.exit(exit_code)

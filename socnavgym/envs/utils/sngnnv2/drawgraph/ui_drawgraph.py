# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'drawGraph.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_SocNavWidget(object):
    def setupUi(self, SocNavWidget):
        if not SocNavWidget.objectName():
            SocNavWidget.setObjectName(u"SocNavWidget")
        SocNavWidget.resize(640, 557)
        self.horizontalLayout = QHBoxLayout(SocNavWidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.widget = QWidget(SocNavWidget)
        self.widget.setObjectName(u"widget")
        self.widget.setMinimumSize(QSize(200, 0))

        self.horizontalLayout.addWidget(self.widget)

        self.tableWidget = QTableWidget(SocNavWidget)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setMinimumSize(QSize(400, 0))
        self.tableWidget.setStyleSheet(u"font: 15pt \"Ubuntu\";")

        self.horizontalLayout.addWidget(self.tableWidget)


        self.retranslateUi(SocNavWidget)

        QMetaObject.connectSlotsByName(SocNavWidget)
    # setupUi

    def retranslateUi(self, SocNavWidget):
        SocNavWidget.setWindowTitle(QCoreApplication.translate("SocNavWidget", u"SocNav graph inspector", None))
    # retranslateUi


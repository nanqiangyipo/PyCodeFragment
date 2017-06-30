# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tantan.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tantan = QtWidgets.QPushButton(self.centralwidget)
        self.tantan.setGeometry(QtCore.QRect(300, 260, 191, 81))
        self.tantan.setAutoDefault(True)
        self.tantan.setObjectName("tantan")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tantan.clicked.connect(self.myslot)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tantan.setText(_translate("MainWindow", "弹弹"))

    def myslot(self):
        QtWidgets.QMessageBox.information(self.tantan, "标题", "这是第一个PyQt5 GUI程序")

if __name__=='__main__':
    app = QApplication(sys.argv)
    mainwin = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainwin)
    mainwin.show()
    sys.exit(app.exec_())

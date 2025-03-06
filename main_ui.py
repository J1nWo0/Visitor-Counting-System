# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'sample.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1500, 800)
        MainWindow.setMinimumSize(QSize(1500, 800))
        MainWindow.setMaximumSize(QSize(1500, 800))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(410, 30, 1031, 571))
        font = QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.start_btn = QPushButton(self.centralwidget)
        self.start_btn.setObjectName(u"start_btn")
        self.start_btn.setGeometry(QRect(630, 630, 261, 51))
        font1 = QFont()
        font1.setPointSize(12)
        font1.setBold(True)
        self.start_btn.setFont(font1)
        self.stop_btn = QPushButton(self.centralwidget)
        self.stop_btn.setObjectName(u"stop_btn")
        self.stop_btn.setGeometry(QRect(950, 630, 261, 51))
        self.stop_btn.setFont(font1)
        self.cap_1 = QLabel(self.centralwidget)
        self.cap_1.setObjectName(u"cap_1")
        self.cap_1.setGeometry(QRect(50, 70, 221, 191))
        self.cap_2 = QLabel(self.centralwidget)
        self.cap_2.setObjectName(u"cap_2")
        self.cap_2.setGeometry(QRect(50, 300, 221, 191))
        self.cap_3 = QLabel(self.centralwidget)
        self.cap_3.setObjectName(u"cap_3")
        self.cap_3.setGeometry(QRect(50, 530, 221, 191))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText("")
        self.start_btn.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.stop_btn.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.cap_1.setText("")
        self.cap_2.setText("")
        self.cap_3.setText("")
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QSize(640, 480))
        MainWindow.setAnimated(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.oscPort = QLineEdit(self.centralwidget)
        self.oscPort.setObjectName(u"oscPort")
        self.oscPort.setMinimumSize(QSize(0, 0))
        self.oscPort.setMaximumSize(QSize(60, 16777215))
        self.oscPort.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.oscPort, 2, 4, 1, 1)

        self.camLabel = QLabel(self.centralwidget)
        self.camLabel.setObjectName(u"camLabel")
        sizePolicy.setHeightForWidth(self.camLabel.sizePolicy().hasHeightForWidth())
        self.camLabel.setSizePolicy(sizePolicy)
        self.camLabel.setSizeIncrement(QSize(1, 1))
        self.camLabel.setFrameShape(QFrame.Shape.StyledPanel)
        self.camLabel.setScaledContents(False)
        self.camLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camLabel.setWordWrap(False)
        self.camLabel.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout.addWidget(self.camLabel, 0, 0, 1, 5)

        self.stopButton = QPushButton(self.centralwidget)
        self.stopButton.setObjectName(u"stopButton")

        self.gridLayout.addWidget(self.stopButton, 1, 1, 2, 1)

        self.oscPortLabel = QLabel(self.centralwidget)
        self.oscPortLabel.setObjectName(u"oscPortLabel")
        font = QFont()
        font.setBold(True)
        self.oscPortLabel.setFont(font)
        self.oscPortLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oscPortLabel.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout.addWidget(self.oscPortLabel, 1, 4, 1, 1)

        self.oscAddress = QLineEdit(self.centralwidget)
        self.oscAddress.setObjectName(u"oscAddress")

        self.gridLayout.addWidget(self.oscAddress, 2, 3, 1, 1)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")

        self.gridLayout.addWidget(self.startButton, 1, 0, 2, 1)

        self.oscAddressLabel = QLabel(self.centralwidget)
        self.oscAddressLabel.setObjectName(u"oscAddressLabel")
        self.oscAddressLabel.setFont(font)
        self.oscAddressLabel.setTextFormat(Qt.TextFormat.AutoText)
        self.oscAddressLabel.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout.addWidget(self.oscAddressLabel, 1, 3, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 2, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Mediapipe", None))
        self.oscPort.setText(QCoreApplication.translate("MainWindow", u"8000", None))
        self.camLabel.setText(QCoreApplication.translate("MainWindow", u"press START", None))
        self.stopButton.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.oscPortLabel.setText(QCoreApplication.translate("MainWindow", u"Port", None))
        self.oscAddress.setText(QCoreApplication.translate("MainWindow", u"127.0.0.1", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.oscAddressLabel.setText(QCoreApplication.translate("MainWindow", u"IP Address", None))
    # retranslateUi


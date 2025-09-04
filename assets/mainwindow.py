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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1100, 700)
        MainWindow.setMinimumSize(QSize(1100, 700))
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
        self.stopButton = QPushButton(self.centralwidget)
        self.stopButton.setObjectName(u"stopButton")

        self.gridLayout.addWidget(self.stopButton, 2, 2, 1, 1)

        self.oscPort = QLineEdit(self.centralwidget)
        self.oscPort.setObjectName(u"oscPort")
        self.oscPort.setMinimumSize(QSize(0, 0))
        self.oscPort.setMaximumSize(QSize(60, 16777215))
        self.oscPort.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.oscPort, 2, 7, 1, 1)

        self.oscPortLabel = QLabel(self.centralwidget)
        self.oscPortLabel.setObjectName(u"oscPortLabel")
        font = QFont()
        font.setBold(True)
        self.oscPortLabel.setFont(font)
        self.oscPortLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oscPortLabel.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout.addWidget(self.oscPortLabel, 1, 7, 1, 1)

        self.oscAddress = QLineEdit(self.centralwidget)
        self.oscAddress.setObjectName(u"oscAddress")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.oscAddress.sizePolicy().hasHeightForWidth())
        self.oscAddress.setSizePolicy(sizePolicy1)
        self.oscAddress.setMaximumSize(QSize(300, 16777215))

        self.gridLayout.addWidget(self.oscAddress, 2, 6, 1, 1)

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

        self.gridLayout.addWidget(self.camLabel, 0, 0, 1, 8)

        self.oscAddressLabel = QLabel(self.centralwidget)
        self.oscAddressLabel.setObjectName(u"oscAddressLabel")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.oscAddressLabel.sizePolicy().hasHeightForWidth())
        self.oscAddressLabel.setSizePolicy(sizePolicy2)
        self.oscAddressLabel.setFont(font)
        self.oscAddressLabel.setTextFormat(Qt.TextFormat.AutoText)
        self.oscAddressLabel.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout.addWidget(self.oscAddressLabel, 1, 6, 1, 1)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")

        self.gridLayout.addWidget(self.startButton, 2, 1, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 2, 3, 1, 1)

        self.horizontalSpacer = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 2, 5, 1, 1)

        self.modelSelection = QComboBox(self.centralwidget)
        self.modelSelection.addItem("")
        self.modelSelection.addItem("")
        self.modelSelection.addItem("")
        self.modelSelection.addItem("")
        self.modelSelection.addItem("")
        self.modelSelection.addItem("")
        self.modelSelection.setObjectName(u"modelSelection")
        sizePolicy1.setHeightForWidth(self.modelSelection.sizePolicy().hasHeightForWidth())
        self.modelSelection.setSizePolicy(sizePolicy1)
        self.modelSelection.setMinimumSize(QSize(260, 0))

        self.gridLayout.addWidget(self.modelSelection, 2, 4, 1, 1)

        self.oscAddressLabel_2 = QLabel(self.centralwidget)
        self.oscAddressLabel_2.setObjectName(u"oscAddressLabel_2")
        self.oscAddressLabel_2.setFont(font)
        self.oscAddressLabel_2.setTextFormat(Qt.TextFormat.AutoText)
        self.oscAddressLabel_2.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout.addWidget(self.oscAddressLabel_2, 1, 4, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Mediapipe", None))
        self.stopButton.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.oscPort.setText(QCoreApplication.translate("MainWindow", u"8000", None))
        self.oscPortLabel.setText(QCoreApplication.translate("MainWindow", u"Port", None))
        self.oscAddress.setText(QCoreApplication.translate("MainWindow", u"127.0.0.1", None))
        self.camLabel.setText(QCoreApplication.translate("MainWindow", u"press START", None))
        self.oscAddressLabel.setText(QCoreApplication.translate("MainWindow", u"IP Address", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.modelSelection.setItemText(0, QCoreApplication.translate("MainWindow", u"Hand landmark detection", u"hand_lm"))
        self.modelSelection.setItemText(1, QCoreApplication.translate("MainWindow", u"Gesture recognition", u"gesture"))
        self.modelSelection.setItemText(2, QCoreApplication.translate("MainWindow", u"Pose landmark detection", u"pose"))
        self.modelSelection.setItemText(3, QCoreApplication.translate("MainWindow", u"Face detection", u"face"))
        self.modelSelection.setItemText(4, QCoreApplication.translate("MainWindow", u"Face landmark detection", u"face_lm"))
        self.modelSelection.setItemText(5, QCoreApplication.translate("MainWindow", u"Object detection", u"object"))

        self.oscAddressLabel_2.setText(QCoreApplication.translate("MainWindow", u"Mediapipe solution", None))
    # retranslateUi


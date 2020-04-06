from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainLoader(object):
    def setupUi(self, MainLoader):
        MainLoader.setObjectName("MainLoader")
        MainLoader.resize(600, 275)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainLoader.sizePolicy().hasHeightForWidth())
        MainLoader.setSizePolicy(sizePolicy)
        MainLoader.setMinimumSize(QtCore.QSize(600, 275))
        MainLoader.setMaximumSize(QtCore.QSize(16777215, 275))
        self.loaderWidget = QtWidgets.QWidget(MainLoader)
        self.loaderWidget.setObjectName("loaderWidget")
        self.loaderLayout = QtWidgets.QVBoxLayout(self.loaderWidget)
        self.loaderLayout.setContentsMargins(10, 10, 10, 10)
        self.loaderLayout.setSpacing(10)
        self.loaderLayout.setObjectName("loaderLayout")
        self.videoGroup = QtWidgets.QGroupBox(self.loaderWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.videoGroup.setFont(font)
        self.videoGroup.setObjectName("videoGroup")
        self.videoLayout = QtWidgets.QHBoxLayout(self.videoGroup)
        self.videoLayout.setContentsMargins(10, 10, 10, 10)
        self.videoLayout.setSpacing(5)
        self.videoLayout.setObjectName("videoLayout")
        self.videoPath = QtWidgets.QLineEdit(self.videoGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.videoPath.setFont(font)
        self.videoPath.setObjectName("videoPath")
        self.videoLayout.addWidget(self.videoPath)
        self.videoButton = QtWidgets.QPushButton(self.videoGroup)
        self.videoButton.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.videoButton.setFont(font)
        self.videoButton.setObjectName("videoButton")
        self.videoLayout.addWidget(self.videoButton)
        self.loaderLayout.addWidget(self.videoGroup)
        self.LabelGroup = QtWidgets.QGroupBox(self.loaderWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.LabelGroup.setFont(font)
        self.LabelGroup.setObjectName("LabelGroup")
        self.labelLayout = QtWidgets.QHBoxLayout(self.LabelGroup)
        self.labelLayout.setContentsMargins(10, 10, 10, 10)
        self.labelLayout.setSpacing(5)
        self.labelLayout.setObjectName("labelLayout")
        self.labelPath = QtWidgets.QLineEdit(self.LabelGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.labelPath.setFont(font)
        self.labelPath.setObjectName("labelPath")
        self.labelLayout.addWidget(self.labelPath)
        self.labelButton = QtWidgets.QPushButton(self.LabelGroup)
        self.labelButton.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.labelButton.setFont(font)
        self.labelButton.setObjectName("labelButton")
        self.labelLayout.addWidget(self.labelButton)
        self.loaderLayout.addWidget(self.LabelGroup)
        self.upperLine = QtWidgets.QFrame(self.loaderWidget)
        self.upperLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.upperLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.upperLine.setObjectName("upperLine")
        self.loaderLayout.addWidget(self.upperLine)
        self.outputGroup = QtWidgets.QGroupBox(self.loaderWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.outputGroup.setFont(font)
        self.outputGroup.setObjectName("outputGroup")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.outputGroup)
        self.horizontalLayout.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.outputPath = QtWidgets.QLineEdit(self.outputGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.outputPath.setFont(font)
        self.outputPath.setObjectName("outputPath")
        self.horizontalLayout.addWidget(self.outputPath)
        self.outputButton = QtWidgets.QPushButton(self.outputGroup)
        self.outputButton.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.outputButton.setFont(font)
        self.outputButton.setObjectName("outputButton")
        self.horizontalLayout.addWidget(self.outputButton)
        self.loaderLayout.addWidget(self.outputGroup)
        self.lowerLine = QtWidgets.QFrame(self.loaderWidget)
        self.lowerLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.lowerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lowerLine.setObjectName("lowerLine")
        self.loaderLayout.addWidget(self.lowerLine)
        self.controlLayout = QtWidgets.QHBoxLayout()
        self.controlLayout.setContentsMargins(0, 0, 0, 0)
        self.controlLayout.setSpacing(50)
        self.controlLayout.setObjectName("controlLayout")
        self.loadButton = QtWidgets.QPushButton(self.loaderWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.loadButton.setFont(font)
        self.loadButton.setObjectName("loadButton")
        self.controlLayout.addWidget(self.loadButton)
        self.exitButton = QtWidgets.QPushButton(self.loaderWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.exitButton.setFont(font)
        self.exitButton.setObjectName("exitButton")
        self.controlLayout.addWidget(self.exitButton)
        self.loaderLayout.addLayout(self.controlLayout)
        MainLoader.setCentralWidget(self.loaderWidget)

        self.retranslateUi(MainLoader)
        QtCore.QMetaObject.connectSlotsByName(MainLoader)

    def retranslateUi(self, MainLoader):
        _translate = QtCore.QCoreApplication.translate
        MainLoader.setWindowTitle(_translate("MainLoader", "Manual Labelling File Loader"))
        self.videoGroup.setTitle(_translate("MainLoader", "Select Video"))
        self.videoButton.setText(_translate("MainLoader", "Choose Video"))
        self.LabelGroup.setTitle(_translate("MainLoader", "Select Label List"))
        self.labelButton.setText(_translate("MainLoader", "Choose List"))
        self.outputGroup.setTitle(_translate("MainLoader", "Output Directory"))
        self.outputButton.setText(_translate("MainLoader", "Choose Directory"))
        self.loadButton.setText(_translate("MainLoader", "Load"))
        self.exitButton.setText(_translate("MainLoader", "Exit"))


class Ui_ControlViewer(object):
    def setupUi(self, ControlViewer):
        ControlViewer.setObjectName("ControlViewer")
        ControlViewer.resize(587, 330)
        ControlViewer.setMinimumSize(QtCore.QSize(587, 330))
        ControlViewer.setMaximumSize(QtCore.QSize(589, 330))
        self.frmctrlWidget = QtWidgets.QWidget(ControlViewer)
        self.frmctrlWidget.setObjectName("frmctrlWidget")
        self.frmctrlLayout = QtWidgets.QVBoxLayout(self.frmctrlWidget)
        self.frmctrlLayout.setContentsMargins(10, 10, 10, 10)
        self.frmctrlLayout.setSpacing(10)
        self.frmctrlLayout.setObjectName("frmctrlLayout")
        self.frmsldrLayout = QtWidgets.QHBoxLayout()
        self.frmsldrLayout.setSpacing(10)
        self.frmsldrLayout.setObjectName("frmsldrLayout")
        self.frameSlider = QtWidgets.QSlider(self.frmctrlWidget)
        self.frameSlider.setMinimumSize(QtCore.QSize(0, 25))
        self.frameSlider.setMaximumSize(QtCore.QSize(16777215, 25))
        self.frameSlider.setOrientation(QtCore.Qt.Horizontal)
        self.frameSlider.setObjectName("frameSlider")
        self.frmsldrLayout.addWidget(self.frameSlider)
        self.sliderValue = QtWidgets.QSpinBox(self.frmctrlWidget)
        self.sliderValue.setMinimumSize(QtCore.QSize(50, 0))
        self.sliderValue.setObjectName("sliderValue")
        self.frmsldrLayout.addWidget(self.sliderValue)
        self.frmctrlLayout.addLayout(self.frmsldrLayout)
        self.subctrlLayout = QtWidgets.QHBoxLayout()
        self.subctrlLayout.setSpacing(15)
        self.subctrlLayout.setObjectName("subctrlLayout")
        self.prevButton = QtWidgets.QPushButton(self.frmctrlWidget)
        self.prevButton.setMinimumSize(QtCore.QSize(0, 190))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.prevButton.setFont(font)
        self.prevButton.setObjectName("prevButton")
        self.subctrlLayout.addWidget(self.prevButton)
        self.frameGroup = QtWidgets.QGroupBox(self.frmctrlWidget)
        self.frameGroup.setMinimumSize(QtCore.QSize(120, 191))
        self.frameGroup.setMaximumSize(QtCore.QSize(16777215, 191))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.frameGroup.setFont(font)
        self.frameGroup.setObjectName("frameGroup")
        self.frameLayout = QtWidgets.QVBoxLayout(self.frameGroup)
        self.frameLayout.setContentsMargins(10, 0, 10, 10)
        self.frameLayout.setSpacing(5)
        self.frameLayout.setObjectName("frameLayout")
        self.frmintLayout = QtWidgets.QVBoxLayout()
        self.frmintLayout.setSpacing(0)
        self.frmintLayout.setObjectName("frmintLayout")
        self.labelInitial = QtWidgets.QLabel(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.labelInitial.setFont(font)
        self.labelInitial.setObjectName("labelInitial")
        self.frmintLayout.addWidget(self.labelInitial)
        self.frameInitial = QtWidgets.QSpinBox(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frameInitial.setFont(font)
        self.frameInitial.setMaximum(16777215)
        self.frameInitial.setObjectName("frameInitial")
        self.frmintLayout.addWidget(self.frameInitial)
        self.frameLayout.addLayout(self.frmintLayout)
        self.frmstpLayout = QtWidgets.QVBoxLayout()
        self.frmstpLayout.setSpacing(0)
        self.frmstpLayout.setObjectName("frmstpLayout")
        self.labelStep = QtWidgets.QLabel(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.labelStep.setFont(font)
        self.labelStep.setObjectName("labelStep")
        self.frmstpLayout.addWidget(self.labelStep)
        self.frameStep = QtWidgets.QSpinBox(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frameStep.setFont(font)
        self.frameStep.setMinimum(1)
        self.frameStep.setMaximum(16777215)
        self.frameStep.setObjectName("frameStep")
        self.frmstpLayout.addWidget(self.frameStep)
        self.frameLayout.addLayout(self.frmstpLayout)
        self.frmfinLayout = QtWidgets.QVBoxLayout()
        self.frmfinLayout.setSpacing(0)
        self.frmfinLayout.setObjectName("frmfinLayout")
        self.labelFinal = QtWidgets.QLabel(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.labelFinal.setFont(font)
        self.labelFinal.setObjectName("labelFinal")
        self.frmfinLayout.addWidget(self.labelFinal)
        self.frameFinal = QtWidgets.QSpinBox(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frameFinal.setFont(font)
        self.frameFinal.setMinimum(1)
        self.frameFinal.setMaximum(16777215)
        self.frameFinal.setObjectName("frameFinal")
        self.frmfinLayout.addWidget(self.frameFinal)
        self.frameLayout.addLayout(self.frmfinLayout)
        self.frmselLayout = QtWidgets.QVBoxLayout()
        self.frmselLayout.setContentsMargins(-1, 2, -1, -1)
        self.frmselLayout.setSpacing(5)
        self.frmselLayout.setObjectName("frmselLayout")
        self.frmselLine = QtWidgets.QFrame(self.frameGroup)
        self.frmselLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.frmselLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frmselLine.setObjectName("frmselLine")
        self.frmselLayout.addWidget(self.frmselLine)
        self.frmselButton = QtWidgets.QPushButton(self.frameGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frmselButton.setFont(font)
        self.frmselButton.setObjectName("frmselButton")
        self.frmselLayout.addWidget(self.frmselButton)
        self.frameLayout.addLayout(self.frmselLayout)
        self.subctrlLayout.addWidget(self.frameGroup)
        self.labellingLayout = QtWidgets.QVBoxLayout()
        self.labellingLayout.setSpacing(5)
        self.labellingLayout.setObjectName("labellingLayout")
        self.labelGroup = QtWidgets.QGroupBox(self.frmctrlWidget)
        self.labelGroup.setMinimumSize(QtCore.QSize(250, 60))
        self.labelGroup.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelGroup.setFont(font)
        self.labelGroup.setObjectName("labelGroup")
        self.labelLayout = QtWidgets.QGridLayout(self.labelGroup)
        self.labelLayout.setContentsMargins(10, 0, 10, 10)
        self.labelLayout.setHorizontalSpacing(10)
        self.labelLayout.setVerticalSpacing(0)
        self.labelLayout.setObjectName("labelLayout")
        self.idBox = QtWidgets.QComboBox(self.labelGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.idBox.setFont(font)
        self.idBox.setObjectName("idBox")
        self.idBox.addItem("")
        self.labelLayout.addWidget(self.idBox, 1, 1, 1, 1)
        self.modeLabel = QtWidgets.QLabel(self.labelGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.modeLabel.setFont(font)
        self.modeLabel.setObjectName("modeLabel")
        self.labelLayout.addWidget(self.modeLabel, 0, 0, 1, 1)
        self.idLabel = QtWidgets.QLabel(self.labelGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.idLabel.setFont(font)
        self.idLabel.setObjectName("idLabel")
        self.labelLayout.addWidget(self.idLabel, 0, 1, 1, 1)
        self.modeBox = QtWidgets.QComboBox(self.labelGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.modeBox.setFont(font)
        self.modeBox.setObjectName("modeBox")
        self.modeBox.addItem("")
        self.modeBox.addItem("")
        self.labelLayout.addWidget(self.modeBox, 1, 0, 1, 1)
        self.labellingLayout.addWidget(self.labelGroup)
        self.coordinateGroup = QtWidgets.QGroupBox(self.frmctrlWidget)
        self.coordinateGroup.setMinimumSize(QtCore.QSize(250, 60))
        self.coordinateGroup.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.coordinateGroup.setFont(font)
        self.coordinateGroup.setObjectName("coordinateGroup")
        self.coordinateLayout = QtWidgets.QGridLayout(self.coordinateGroup)
        self.coordinateLayout.setContentsMargins(10, 0, 10, 10)
        self.coordinateLayout.setHorizontalSpacing(10)
        self.coordinateLayout.setVerticalSpacing(0)
        self.coordinateLayout.setObjectName("coordinateLayout")
        self.xLabel = QtWidgets.QLabel(self.coordinateGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.xLabel.setFont(font)
        self.xLabel.setObjectName("xLabel")
        self.coordinateLayout.addWidget(self.xLabel, 0, 0, 1, 1)
        self.yLabel = QtWidgets.QLabel(self.coordinateGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.yLabel.setFont(font)
        self.yLabel.setObjectName("yLabel")
        self.coordinateLayout.addWidget(self.yLabel, 0, 1, 1, 1)
        self.xValue = QtWidgets.QSpinBox(self.coordinateGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.xValue.setFont(font)
        self.xValue.setMinimum(-1)
        self.xValue.setObjectName("xValue")
        self.coordinateLayout.addWidget(self.xValue, 1, 0, 1, 1)
        self.yValue = QtWidgets.QSpinBox(self.coordinateGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.yValue.setFont(font)
        self.yValue.setMinimum(-1)
        self.yValue.setObjectName("yValue")
        self.coordinateLayout.addWidget(self.yValue, 1, 1, 1, 1)
        self.labellingLayout.addWidget(self.coordinateGroup)
        self.sizeGroup = QtWidgets.QGroupBox(self.frmctrlWidget)
        self.sizeGroup.setMinimumSize(QtCore.QSize(250, 60))
        self.sizeGroup.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.sizeGroup.setFont(font)
        self.sizeGroup.setObjectName("sizeGroup")
        self.sizeLayout = QtWidgets.QGridLayout(self.sizeGroup)
        self.sizeLayout.setContentsMargins(10, 0, 10, 10)
        self.sizeLayout.setHorizontalSpacing(10)
        self.sizeLayout.setVerticalSpacing(0)
        self.sizeLayout.setObjectName("sizeLayout")
        self.wLabel = QtWidgets.QLabel(self.sizeGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.wLabel.setFont(font)
        self.wLabel.setObjectName("wLabel")
        self.sizeLayout.addWidget(self.wLabel, 0, 0, 1, 1)
        self.hLabel = QtWidgets.QLabel(self.sizeGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.hLabel.setFont(font)
        self.hLabel.setObjectName("hLabel")
        self.sizeLayout.addWidget(self.hLabel, 0, 1, 1, 1)
        self.wValue = QtWidgets.QSpinBox(self.sizeGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.wValue.setFont(font)
        self.wValue.setObjectName("wValue")
        self.sizeLayout.addWidget(self.wValue, 1, 0, 1, 1)
        self.hValue = QtWidgets.QSpinBox(self.sizeGroup)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.hValue.setFont(font)
        self.hValue.setObjectName("hValue")
        self.sizeLayout.addWidget(self.hValue, 1, 1, 1, 1)
        self.labellingLayout.addWidget(self.sizeGroup)
        self.subctrlLayout.addLayout(self.labellingLayout)
        self.nextButton = QtWidgets.QPushButton(self.frmctrlWidget)
        self.nextButton.setMinimumSize(QtCore.QSize(0, 190))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.nextButton.setFont(font)
        self.nextButton.setObjectName("nextButton")
        self.subctrlLayout.addWidget(self.nextButton)
        self.frmctrlLayout.addLayout(self.subctrlLayout)
        self.progctrlLayout = QtWidgets.QHBoxLayout()
        self.progctrlLayout.setSpacing(10)
        self.progctrlLayout.setObjectName("progctrlLayout")
        self.previewButton = QtWidgets.QRadioButton(self.frmctrlWidget)
        self.previewButton.setMinimumSize(QtCore.QSize(90, 0))
        self.previewButton.setChecked(True)
        self.previewButton.setObjectName("previewButton")
        self.savemodeGroup = QtWidgets.QButtonGroup(ControlViewer)
        self.savemodeGroup.setObjectName("savemodeGroup")
        self.savemodeGroup.addButton(self.previewButton)
        self.progctrlLayout.addWidget(self.previewButton)
        self.recordButton = QtWidgets.QRadioButton(self.frmctrlWidget)
        self.recordButton.setMinimumSize(QtCore.QSize(90, 0))
        self.recordButton.setObjectName("recordButton")
        self.savemodeGroup.addButton(self.recordButton)
        self.progctrlLayout.addWidget(self.recordButton)
        spacerItem = QtWidgets.QSpacerItem(40, 25, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.progctrlLayout.addItem(spacerItem)
        self.exitButton = QtWidgets.QPushButton(self.frmctrlWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.exitButton.setFont(font)
        self.exitButton.setObjectName("exitButton")
        self.progctrlLayout.addWidget(self.exitButton)
        self.frmctrlLayout.addLayout(self.progctrlLayout)
        ControlViewer.setCentralWidget(self.frmctrlWidget)
        self.statusBar = QtWidgets.QStatusBar(ControlViewer)
        self.statusBar.setObjectName("statusBar")
        ControlViewer.setStatusBar(self.statusBar)
        self.menuBar = QtWidgets.QMenuBar(ControlViewer)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 587, 21))
        self.menuBar.setObjectName("menuBar")
        ControlViewer.setMenuBar(self.menuBar)

        self.retranslateUi(ControlViewer)
        self.frameSlider.valueChanged['int'].connect(self.sliderValue.setValue)
        self.sliderValue.valueChanged['int'].connect(self.frameSlider.setValue)
        self.prevButton.clicked.connect(self.sliderValue.stepDown)
        self.nextButton.clicked.connect(self.sliderValue.stepUp)
        QtCore.QMetaObject.connectSlotsByName(ControlViewer)

    def retranslateUi(self, ControlViewer):
        _translate = QtCore.QCoreApplication.translate
        ControlViewer.setWindowTitle(_translate("ControlViewer", "Manual Labelling Control"))
        self.prevButton.setText(_translate("ControlViewer", "◀"))
        self.frameGroup.setTitle(_translate("ControlViewer", "Frame Setup"))
        self.labelInitial.setText(_translate("ControlViewer", "Initial Frame"))
        self.labelStep.setText(_translate("ControlViewer", "Frame Step"))
        self.labelFinal.setText(_translate("ControlViewer", "Final Frame"))
        self.frmselButton.setText(_translate("ControlViewer", "Update"))
        self.labelGroup.setTitle(_translate("ControlViewer", "Label"))
        self.idBox.setItemText(0, _translate("ControlViewer", "[None]"))
        self.modeLabel.setText(_translate("ControlViewer", "Mode"))
        self.idLabel.setText(_translate("ControlViewer", "Name"))
        self.modeBox.setItemText(0, _translate("ControlViewer", "Fixed"))
        self.modeBox.setItemText(1, _translate("ControlViewer", "Dynamic"))
        self.coordinateGroup.setTitle(_translate("ControlViewer", "Coordinates"))
        self.xLabel.setText(_translate("ControlViewer", "X value"))
        self.yLabel.setText(_translate("ControlViewer", "Y value"))
        self.sizeGroup.setTitle(_translate("ControlViewer", "Size"))
        self.wLabel.setText(_translate("ControlViewer", "Width"))
        self.hLabel.setText(_translate("ControlViewer", "Height"))
        self.nextButton.setText(_translate("ControlViewer", "▶"))
        self.previewButton.setText(_translate("ControlViewer", "Preview Mode"))
        self.recordButton.setText(_translate("ControlViewer", "Record Mode"))
        self.exitButton.setText(_translate("ControlViewer", "Exit"))


class Ui_FrameViewer(object):
    def setupUi(self, FrameViewer):
        FrameViewer.setObjectName("FrameViewer")
        FrameViewer.resize(100, 120)
        self.frameWidget = QtWidgets.QWidget(FrameViewer)
        self.frameWidget.setObjectName("frameWidget")
        self.frameLayout = QtWidgets.QGridLayout(self.frameWidget)
        self.frameLayout.setContentsMargins(0, 0, 0, 0)
        self.frameLayout.setSpacing(0)
        self.frameLayout.setObjectName("frameLayout")
        self.frameDisplay = QtWidgets.QGraphicsView(self.frameWidget)
        self.frameDisplay.setObjectName("frameDisplay")
        self.frameLayout.addWidget(self.frameDisplay, 0, 0, 1, 1)
        FrameViewer.setCentralWidget(self.frameWidget)
        self.statusBar = QtWidgets.QStatusBar(FrameViewer)
        self.statusBar.setObjectName("statusBar")
        FrameViewer.setStatusBar(self.statusBar)

        self.retranslateUi(FrameViewer)
        QtCore.QMetaObject.connectSlotsByName(FrameViewer)

    def retranslateUi(self, FrameViewer):
        _translate = QtCore.QCoreApplication.translate
        FrameViewer.setWindowTitle(_translate("FrameViewer", "Frame View"))


class Ui_ListViewer(object):
    def setupUi(self, ListViewer):
        ListViewer.setObjectName("ListViewer")
        ListViewer.resize(215, 500)
        ListViewer.setMinimumSize(QtCore.QSize(215, 200))
        self.listWidget = QtWidgets.QWidget(ListViewer)
        self.listWidget.setObjectName("listWidget")
        self.listLayout = QtWidgets.QVBoxLayout(self.listWidget)
        self.listLayout.setContentsMargins(0, 0, 0, 0)
        self.listLayout.setSpacing(0)
        self.listLayout.setObjectName("listLayout")
        self.lblpreButton = QtWidgets.QPushButton(self.listWidget)
        self.lblpreButton.setMinimumSize(QtCore.QSize(0, 25))
        self.lblpreButton.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.lblpreButton.setFont(font)
        self.lblpreButton.setObjectName("lblpreButton")
        self.listLayout.addWidget(self.lblpreButton)
        self.listTable = QtWidgets.QTableWidget(self.listWidget)
        self.listTable.setEnabled(False)
        self.listTable.setMinimumSize(QtCore.QSize(215, 0))
        self.listTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.listTable.setObjectName("listTable")
        self.listTable.setColumnCount(5)
        self.listTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.listTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.listTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.listTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.listTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.listTable.setHorizontalHeaderItem(4, item)
        self.listTable.horizontalHeader().setDefaultSectionSize(40)
        self.listLayout.addWidget(self.listTable)
        self.lblnxtButton = QtWidgets.QPushButton(self.listWidget)
        self.lblnxtButton.setMinimumSize(QtCore.QSize(0, 25))
        self.lblnxtButton.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.lblnxtButton.setFont(font)
        self.lblnxtButton.setObjectName("lblnxtButton")
        self.listLayout.addWidget(self.lblnxtButton)
        ListViewer.setCentralWidget(self.listWidget)

        self.retranslateUi(ListViewer)
        QtCore.QMetaObject.connectSlotsByName(ListViewer)

    def retranslateUi(self, ListViewer):
        _translate = QtCore.QCoreApplication.translate
        ListViewer.setWindowTitle(_translate("ListViewer", "List Viewer"))
        self.lblpreButton.setText(_translate("ListViewer", "▲"))
        item = self.listTable.horizontalHeaderItem(0)
        item.setText(_translate("ListViewer", "Name"))
        item = self.listTable.horizontalHeaderItem(1)
        item.setText(_translate("ListViewer", "X"))
        item = self.listTable.horizontalHeaderItem(2)
        item.setText(_translate("ListViewer", "Y"))
        item = self.listTable.horizontalHeaderItem(3)
        item.setText(_translate("ListViewer", "W"))
        item = self.listTable.horizontalHeaderItem(4)
        item.setText(_translate("ListViewer", "H"))
        self.lblnxtButton.setText(_translate("ListViewer", "▼"))
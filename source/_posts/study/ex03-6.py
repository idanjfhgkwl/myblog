## Ex 3-6. 메뉴바 만들기
# https://codetorial.net/pyqt5/basics/menubar.html
# https://doc.qt.io/qt-5/qmenubar.html

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp
from PyQt5.QtGui import QIcon

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        exitAction = QAction(QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        # .png 아이콘과 Exit 라벨을 갖는 하나의 동작(exitAction)을 만들고, 이 동작에 대해 shortcut(단축키)를 정의한다.
        # 메뉴에 마우스를 올렸을 때, 상태바에 나타날 상태팁
        exitAction.triggered.connect(qApp.quit)
        # 이 동작을 선택했을 때,
        # 생성된 시그널 triggered 이 qApp 위젯의 quit() 메소드에 연결되고, 어플리케이션을 종료시킨다.

        self.statusBar()

        menubar = self.menuBar() # 메뉴바 생성
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&File') # File 메뉴 하나 만든다.
        # &는 간편하게 단축키를 설정, F 앞에 앰퍼샌드가 있어서 Alt+F
        filemenu.addAction(exitAction) # exitAction 동작 추가한다.

        self.setWindowTitle('Menubar')
        self.setGeometry(300, 300, 300, 200)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
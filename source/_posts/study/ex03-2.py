## Ex 3-2. 어플리케이션 아이콘 넣기.
# https://codetorial.net/pyqt5/basics/icon.html

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('../image/web.png')) # QIcon() 객체에 이미지
        self.setGeometry(300, 300, 300, 300) # 창의 위치와 크기, x, y 위치, 너비, 높이 -> move() + resize()
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec())
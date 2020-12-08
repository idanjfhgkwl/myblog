## Ex 3-8. 창을 화면의 가운데로
# https://codetorial.net/pyqt5/basics/centering.html

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Centering')
        self.resize(500, 350)
        self.center() # 창을 화면의 가운데로 위치
        self.show()

    def center(self):
        qr = self.frameGeometry() # 창의 위치와 크기 정보를 가져온다.
        cp = QDesktopWidget().availableGeometry().center() # 사용하는 모니터 화면의 가운데 위치를 파악
        qr.moveCenter(cp) # 직사각형 위치를 화면의 중심의 위치로 이동
        self.move(qr.topLeft()) # 현재 창을 화면의 중심으로 이동했던 직사각형 qr의 위치로 이동시킨다.
        # 결과적으로 현재 창의 중심이 화면의 중심과 일치하게 된다.

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
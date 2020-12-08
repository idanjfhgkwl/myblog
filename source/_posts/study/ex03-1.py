## ex 3-1. 창 띄우기
# https://codetorial.net/pyqt5/basics/opening.html

import sys
from PyQt5.QtWidgets import QApplication, QWidget
# 파이큐티 위젯 불러오기
# https://doc.qt.io/qt-5/qtwidgets-index.html

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
    # self -> MyApp 객체

    def initUI(self):
        self.setWindowTitle('My First Application') # 타이틀바 창의 제목
        self.move(300, 300) # 스크린의 해당 위치로 이동
        self.resize(400, 200) # 위젯의 크기 조절
        self.show() # 스크린에 보여줘라
    #

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 모든 PyQt5 어플리케이션은 어플리케이션 객체를 생성해야 한다.
    # https://doc.qt.io/qt-5/qapplication.html
    ex = MyApp()
    sys.exit(app.exec_())
# __name__ == 현재 모듈의 이름이 저장되는 내장 변수
# 여기서는 어떤 .py 코드를 import해서 예제 코드를 수행하지 않았고 직접 실행했기에, __main__이 된다.
# 프로그램이 직접 실행되는 지 / 모듈을 통해 실행되는지 확인한다.
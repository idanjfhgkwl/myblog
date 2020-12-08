## Ex 3-3. 창 닫기.
# https://codetorial.net/pyqt5/basics/closing.html

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QCoreApplication

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        btn = QPushButton('Quit', self)
        # 푸시 버튼 만들기, 첫 번째 파라미터는 버튼에 표시될 텍스트, 두 번째는 버튼이 위치할 부모 위젯(self-MyApp객체)
        # https://codetorial.net/pyqt5/widget/qpushbutton.html
        btn.move(50, 50)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(QCoreApplication.instance().quit)
        # 시그널과 슬롯 https://codetorial.net/pyqt5/signal_slot/index.html
        # 버튼을 클릭하면 clicked 시그널이 만들어진다.
        # instance() 메소드는 현재 인스턴스를 반환한다.
        # quit() 메소드에 연결된다. (어플리케이션 종료)
        # btn 푸시 버튼과 app 어플리케이션 객체 간에 커뮤니케이션 (발신자-수신자, Sender-Receiver)

        self.setWindowTitle('Quit Button')
        self.setGeometry(300, 300, 300, 200)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
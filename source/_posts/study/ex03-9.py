## Ex 3-9. 날짜와 시간 표시하기.
# https://codetorial.net/pyqt5/basics/datetime.html
# http://doc.qt.io/qt-5/qdate.html
# https://doc-snapshots.qt.io/qt5-5.12/qtime.html
# http://doc.qt.io/qt-5/qdatetime.html

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QDate, Qt

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.date = QDate.currentDate()
        self.initUI()

    def initUI(self):
        self.statusBar().showMessage(self.date.toString(Qt.DefaultLocaleLongDate))

        self.setWindowTitle('Date')
        self.setGeometry(300, 300, 400, 200)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())





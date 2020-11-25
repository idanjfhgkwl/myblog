---
title: "PyQt5 Tutorial - PyQt5 기초 (Basics)"
categories:
  - study
output: 
  html_document:
    keep_md: true
---

# PyQt5 기초 (Basics)

## 창 띄우기
출처: [Codetorial](https://codetorial.net/pyqt5/basics/opening.html)

### 예제


```python
## Ex 3-1. 창 띄우기.

import sys
from PyQt5.QtWidgets import QApplication, QWidget


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())
```

### 결과  
![K-20201125-100749](https://user-images.githubusercontent.com/72365720/100169145-185d2980-2f06-11eb-8548-742228d268da.png)  
(Windows7 환경에서 실행)

### 설명


```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget
```

기본적인 UI 구성요소를 제공하는 위젯(클래스)은 `PyQt5.QtWidgets` 모듈에 포함돼 있다. [QtWidgets 공식 문서](https://doc.qt.io/qt-5/qtwidgets-index.html)


```python
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()
```

- `self` MyApp 객체를 말한다.
- `setWindowTitle()` 타이틀바에 나타나는 창의 제목을 설정한다.
- `move()` 위젯을 스크린의 x = 300px, y = 300px의 위치로 이동시킨다.
- `resize()` 위젯의 크기를 너비 400px, 높이 200px로 조절한다.
- `show()` 위젯을 스크린에 보여준다.


```python
if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())
```

**`if __name__ == '__main__':`**  
`__name__`은 현재 모듈의 이름이 저장되는 내장 변수이다. 예를 들어 'test.py'라는 코드를 import해서 예제 코드를 실행하면 `__name__`은 'test'가 된다. 그렇지 않고 코드를 직접 실행한다면 `__name__`은 `__main__`이 된다. 이 한 줄의 코드를 통해 프로그램이 직접 실행되는 지, 모듈을 통해 실행되는 지 확인할 수 있다.  

**`app = QApplication(sys.argv)`**  
모든 PyQt5 어플리케이션은 어플리케이션 객체를 생성해야 한다. [QApplication 공식 문서](https://doc.qt.io/qt-5/qapplication.html)

## 어플리케이션 아이콘 넣기
출처: [Codetorial](https://codetorial.net/pyqt5/basics/icon.html)

### 예제


```python
## Ex 3-2. 어플리케이션 아이콘 넣기.

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('image/web.png'))
        self.setGeometry(300, 300, 300, 200)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
```

### 결과

![K-20201125-102840](https://user-images.githubusercontent.com/72365720/100170477-206a9880-2f09-11eb-90ac-08ae42a02a40.png)

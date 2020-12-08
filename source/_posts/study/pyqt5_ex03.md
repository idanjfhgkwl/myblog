---
title: "PyQt5 Tutorial - PyQt5 기초 (Basics)"
categories:
  - study
output:
  html_document:
    keep_md: true
---

출처: 김민휘, PyQt5 Tutorial - 파이썬으로 만드는 나만의 GUI 프로그램, Codetorial
https://codetorial.net/pyqt5/basics/opening.html

<details markdown="1">
<summary>접기/펼치기</summary>

<!--summary 아래 빈칸 공백 두고 내용을 적는공간-->

# PyQt5 기초 (Basics)

## 창 띄우기
출처: [Codetorial](https://codetorial.net/pyqt5/basics/opening.html)

### 예제


```python
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

### 설명


```python
self.setWindowIcon(QIcon('image/web.png'))
```

**`setWindowIcon()`** 메소드는 어플리케이션 아이콘을 설정하도록 한다.  
이를 위해서 QIcon 객체를 생성했고, QIcon()에 보여질 이미지를 입력한다. (경로 확인)  


```python
self.setGeometry(300, 300, 300, 200)
```

**`setGeometry()`** 메소드는 창의 위치와 크기를 설정한다.  
앞의 두 매개변수는 창의 x, y 위치를 결정하고, 뒤의 두 매개변수는 각각 창의 너비와 높이를 결정한다. 이 메소드는 창 띄우기 예제에서 사용했던 move()와 resize() 메서드를 하나로 합쳐놓은 것과 같다.

## 창 닫기
출처: [Codetorial](https://codetorial.net/pyqt5/basics/closing.html)

### 예제


```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QCoreApplication


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        btn = QPushButton('Quit', self)
        btn.move(50, 50)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(QCoreApplication.instance().quit)

        self.setWindowTitle('Quit Button')
        self.setGeometry(300, 300, 300, 200)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
```

### 설명


```python
btn = QPushButton('Quit', self)
```

푸시버튼을 하나 만든다. 이 btn은 QPushButton 클래스의 인스턴스이다.  
첫번째 파라미터에는 버튼에 표시될 텍스트(Quit)를 입력하고, 두번째 파라미터에는 버튼이 위치할 부모 위젯(self)을 입력한다.


```python
btn.clicked.connect(QCoreApplication.instance().quit)
```

PyQt5에서의 이벤트 처리는 시그널과 슬롯 메커니즘으로 이루어진다. btn을 클릭하면 `clicked` 시그널이 만들어진다.

**`instance()`** 메소드는 현재 인스턴스를 반환한다.

**`clicked`** 시그널은 어플리케이션을 종료하는 quit() 메소드에 연결된다.

이렇게 두 객체 발신자와 수신자(Sender & Receiver) 간에 커뮤니케이션이 이루어잔다. 이 예제에서 발신자는 푸시버튼(btn)이고, 수신자는 어플리케이션 객체(app)이다.

![K-20201126-155725](https://user-images.githubusercontent.com/72365720/100317677-19728180-3000-11eb-916d-1a23c1e65568.png)


## 툴팁 나타내기

</details>
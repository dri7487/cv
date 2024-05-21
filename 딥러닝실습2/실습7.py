#실습7
import cv2 as cv
import numpy as np
import tensorflow as tf
import winsound
import pickle
import sys
from PyQt5.QtWidgets import *

cnn = tf.keras.models.load_model('cnn_for_stanford_dogs.h5')
dog_species = pickle.load(open('dog_species_names.txt', 'rb'))

class DogSpeciesRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('강아지 인식')
        self.setGeometry(200, 200, 700, 100)

        fileButton = QPushButton('강아지 사진 읽기', self)
        recognitionButton = QPushButton('강아지 인식', self)
        quitButton = QPushButton('종료', self)

        fileButton.setGeometry(10, 10, 100, 30)
        recognitionButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)

        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

    def pictureOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, '강아지 사진 읽기', '/')
        self.img = cv.imread(fname[0])
        if self.img is None:
            sys.exit('파일을 찾을 수 없습니다.')
        cv.imshow('Dog image', self.img)

    def recognitionFunction(self):
        x = np.reshape(cv.resize(self.img, (224, 224)), (1, 224, 224, 3))
        res = cnn.predict(x)
        top5 = np.argsort(res[0])[-5:]
        top5_species = [dog_species[i] for i in top5]
        for i in range(5):
            prob = '{:.2f}'.format(res[0][top5[i]] * 100)
            name = str(top5_species[i]).split('.')[-1]
            cv.putText(self.img, prob + '%: ' + name, (10, 100 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow('Dog image', self.img)
        winsound.Beep(1000, 500)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
app.exec_()

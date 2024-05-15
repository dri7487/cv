# 실습6 수정
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.reshape(50000, 3072)  # 이미지를 평탄화
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype(np.float32) / 255.0  # 정규화
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 원-핫 인코딩
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 심층 신경망 모델 구성
dmlp = Sequential()
dmlp.add(Dense(units=1024, activation='relu', input_shape=(3072,)))
dmlp.add(BatchNormalization())
dmlp.add(Dropout(0.5))
dmlp.add(Dense(units=512, activation='relu'))
dmlp.add(BatchNormalization())
dmlp.add(Dropout(0.5))
dmlp.add(Dense(units=512, activation='relu'))
dmlp.add(BatchNormalization())
dmlp.add(Dropout(0.5))
dmlp.add(Dense(units=10, activation='softmax'))

# 모델 컴파일
dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# 모델 훈련
hist = dmlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

# 모델 평가
accuracy = dmlp.evaluate(x_test, y_test, verbose=0)[1] * 100
print('정확도:', accuracy)

# 정확도 그래프
plt.plot(hist.history['accuracy'], 'r--')
plt.plot(hist.history['val_accuracy'], 'b')
plt.title('Accuracy graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.grid(True)
plt.show()

# 손실 그래프
plt.plot(hist.history['loss'], 'r--')
plt.plot(hist.history['val_loss'], 'b')
plt.title('Loss graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.grid(True)
plt.show()


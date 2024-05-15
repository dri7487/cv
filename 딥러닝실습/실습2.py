# 실습2
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000, 784)  # 이미지 데이터를 784차원 벡터로 변환
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0  # 데이터를 정규화
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 레이블을 원-핫 인코딩
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 모델 구성 및 컴파일
model = Sequential()
model.add(Dense(units=512, activation='tanh', input_shape=(784,)))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='MSE', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

# 모델 평가 및 결과 출력
results = model.evaluate(x_test, y_test, verbose=0)
print('정확도:', results[1] * 100)
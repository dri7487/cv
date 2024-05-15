# 실습1
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# MNIST 데이터셋 시각화
plt.figure(figsize=(24,3))
plt.suptitle('MNIST', fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title(str(y_train[i]), fontsize=30)

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 데이터셋 시각화
plt.figure(figsize=(24,3))
plt.suptitle('CIFAR-10', fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[y_train[i,0]], fontsize=30)
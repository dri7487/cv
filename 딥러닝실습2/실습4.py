#실습4
import tensorflow.keras.datasets as ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(20, 2))
plt.suptitle('First 15 images in the train set')
for i in range(15):
    plt.subplot(1, 15, i + 1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(y_train[i])])
plt.show()

batch_size = 4
generator = ImageDataGenerator(
    rotation_range=20.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
gen = generator.flow(x_train, y_train, batch_size=batch_size)

for a in range(3):
    img, label = gen.next()
    plt.figure(figsize=(8, 2.4))
    plt.suptitle('Generator trial ' + str(a + 1))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(img[i])
        plt.xticks([]); plt.yticks([])
        plt.title(class_names[int(label[i])])
    plt.show()

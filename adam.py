import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.datasets import mnist
from keras.utils import to_categorical

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразование данных в нужный формат и масштабирование
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Преобразование меток в категориальный формат
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Тренировка модели
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(x_test, y_test)

# Вывод результатов
print("Loss:", loss)
print("Accuracy:", accuracy)
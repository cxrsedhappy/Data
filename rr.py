import numpy as np
import tensorflow as tf

# Создание тестовых данных
train_data = np.array([
    [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
    [[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
    [[3], [4], [5], [6], [7], [8], [9], [10], [11], [12]],
    [[4], [5], [6], [7], [8], [9], [10], [11], [12], [13]],
    [[5], [6], [7], [8], [9], [10], [11], [12], [13], [14]]
])

train_labels = np.array([[1], [2], [3], [4], [5], [6]])

# Определение архитектуры рекуррентной сети
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(10, 1)),  # Рекуррентный слой с 64 нейронами
    tf.keras.layers.Dense(1)  # Полносвязный слой с 1 выходным нейроном
])

# Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_data, train_labels, epochs=100)

# Проверка модели
test_data = np.array([[[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]])
predictions = model.predict(test_data)
print(predictions)

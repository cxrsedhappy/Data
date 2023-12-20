import math

class Rosenblatt:
    def __init__(self):
        self.weights = [0.0, 0.0]
        self.threshold = 0.0
        self.learningRate = 0.1
        self.errorThreshold = 0.001

    def train(self, inputs, targets):
        error = 1.0
        while error > self.errorThreshold:
            error = 0.0
            for i in range(len(inputs)):
                sum = 0.0
                for j in range(len(inputs[i])):
                    sum += inputs[i][j] * self.weights[j]
                output = self.activation(sum)
                target = targets[i]
                delta = target - output
                for j in range(len(self.weights)):
                    self.weights[j] += self.learningRate * inputs[i][j] * delta
                self.threshold -= self.learningRate * delta
                error += abs(delta)
            error /= len(inputs)
        print("Обучение завершено!")
        print("Весовые коэффициенты:", self.weights)
        print("Порог:", self.threshold)

    def predict(self, input):
        sum = 0.0
        for i in range(len(input)):
            sum += input[i] * self.weights[i]
        output = self.activation(sum)
        if output > self.threshold:
            return 1
        else:
            return -1

    def predict1(self, input):
        sum = 0.0
        for i in range(len(input)):
            sum += input[i] + self.weights[i]
        output = self.activation(sum)
        if output > self.threshold:
            return 1
        else:
            return -1

    def activation(self, sum):
        if sum > self.threshold:
            return 1.0
        else:
            return -1.0


inputs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
targets = [-1, -1, -1, 1]
neuralNetwork = Rosenblatt()
neuralNetwork.train(inputs, targets)
print("========== И ===========")
testInputs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
for i in range(len(testInputs)):
    prediction = neuralNetwork.predict(testInputs[i])
    print("Вход:", testInputs[i])
    print("Выход:", prediction)
print("========= ИЛИ ==========")
for i in range(len(testInputs)):
    prediction1 = neuralNetwork.predict1(testInputs[i])
    print("Вход:", testInputs[i])
    print("Выход:", prediction1)
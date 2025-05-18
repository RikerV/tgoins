import numpy as np

class Perceptron:
    def __init__(self, num_inputs, num_outputs, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = learning_rate
        self.weights = np.random.rand(num_inputs, num_outputs) - 0.5 # Веса для входов
        self.bias = np.random.rand(num_outputs) - 0.5              # Смещения для каждого нейрона


    def activation_function(self, net_input):
        return (net_input >= 0).astype(int) # Если net >= 0, то 1, иначе 0

    def predict(self, inputs):
        if inputs.shape[0] != self.num_inputs:
            raise ValueError(f"Неверное количество входов. Ожидалось {self.num_inputs}, получено {inputs.shape[0]}")

        net_j = np.dot(inputs, self.weights) + self.bias
        y_j = self.activation_function(net_j)
        return y_j

    def train_single_example(self, inputs, target_outputs):
        predicted_outputs = self.predict(inputs)
        error = target_outputs - predicted_outputs # Вектор ошибок для каждого выходного нейрона

        # Коррекция весов (Дельта-правило / Правило Розенблатта)
        # Δw_ij = η * error_j * x_i
        # Δb_j = η * error_j (так как для bias фиктивный вход x_0 = 1)

        # np.outer(inputs, error) создаст матрицу (num_inputs x num_outputs)
        # где каждый столбец error_j умножается на вектор inputs.
        # Это соответствует (inputs.reshape(-1,1) @ error.reshape(1,-1))
        # или для каждого нейрона j: self.weights[:, j] += self.lr * error[j] * inputs
        # и self.bias[j] += self.lr * error[j]

        # Обновление весов для каждого выходного нейрона
        for j in range(self.num_outputs):
            if error[j] != 0: # Обновляем только если есть ошибка для этого нейрона
                self.weights[:, j] += self.lr * error[j] * inputs
                self.bias[j] += self.lr * error[j] * 1 # *1 для bias

        return error


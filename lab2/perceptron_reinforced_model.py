import numpy as np

class ReinforcedPerceptron:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = (np.random.rand(num_inputs, num_outputs) - 0.5) * 0.2 # Веса для входов в диапазоне [-0.1, 0.1]
        self.bias = (np.random.rand(num_outputs) - 0.5) * 0.2              # Смещения для каждого нейрона в диапазоне [-0.1, 0.1]

    def activation_function(self, net_input):
        return (net_input >= 0).astype(int)

    def predict(self, inputs):
        if inputs.shape[0] != self.num_inputs:
            raise ValueError(f"Неверное количество входов. Ожидалось {self.num_inputs}, получено {inputs.shape[0]}")

        net_j = np.dot(inputs, self.weights) + self.bias
        y_j = self.activation_function(net_j)
        return y_j

    def train_single_example(self, inputs, target_outputs):
        predicted_outputs = self.predict(inputs)
        weights_changed = False

        for j in range(self.num_outputs): # Для каждого выходного нейрона
            if predicted_outputs[j] != target_outputs[j]: # Если выход y_j неправильный
                if target_outputs[j] == 1: # Ожидалась 1, а получили 0 (y_j = 0)
                    # Увеличить веса активных входов: w_ij(t+1) = w_ij(t) + x_i
                    self.weights[:, j] += inputs # inputs это x_i, применяем ко всем весам этого нейрона
                    self.bias[j] += 1            # Увеличить смещение
                    weights_changed = True
                elif target_outputs[j] == 0: # Ожидался 0, а получили 1 (y_j = 1)
                    # Уменьшить веса активных входов: w_ij(t+1) = w_ij(t) - x_i
                    self.weights[:, j] -= inputs
                    self.bias[j] -= 1            # Уменьшить смещение
                    weights_changed = True
        return weights_changed


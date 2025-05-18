import numpy as np

class NeuralNetworkBPA:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.25):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.W1 = np.random.uniform(-0.3, 0.3, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(-0.3, 0.3, (1, self.hidden_size))
        self.W2 = np.random.uniform(-0.3, 0.3, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(-0.3, 0.3, (1, self.output_size))

        # Для хранения активаций во время прямого прохода
        self.activation_hidden = None
        self.activation_output = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def feedforward(self, X):
        # Входной слой -> Скрытый слой
        self.z_hidden = np.dot(X, self.W1) + self.b1
        self.activation_hidden = self._sigmoid(self.z_hidden)

        # Скрытый слой -> Выходной слой
        self.z_output = np.dot(self.activation_hidden, self.W2) + self.b2
        self.activation_output = self._sigmoid(self.z_output)
        return self.activation_output

    def backpropagate(self, X, D):
        # 2. Расчет ошибки и дельт для выходного слоя
        error_output_layer = D - self.activation_output # Ошибка E_k = (d_k - y_k)
        # Дельта для выходного слоя: delta_k = (d_k - y_k) * y_k * (1 - y_k)
        delta_output = error_output_layer * self._sigmoid_derivative(self.activation_output)

        # 3. Расчет ошибки и дельт для скрытого слоя
        # Ошибка скрытого слоя распространяется от выходного: sum(delta_k * w_jk) по k
        error_hidden_layer = np.dot(delta_output, self.W2.T)
        # Дельта для скрытого слоя: delta_j = error_hidden_j * y_j * (1 - y_j)
        delta_hidden = error_hidden_layer * self._sigmoid_derivative(self.activation_hidden)

        # 4. Вычисление градиентов для весов и смещений
        grad_W2 = np.dot(self.activation_hidden.T, delta_output)
        grad_b2 = np.sum(delta_output, axis=0, keepdims=True) # Суммируем по примерам (здесь пример один)

        grad_W1 = np.dot(X.T, delta_hidden)
        grad_b1 = np.sum(delta_hidden, axis=0, keepdims=True) # Суммируем по примерам

        # 5. Обновление весов и смещений
        self.W2 += self.lr * grad_W2
        self.b2 += self.lr * grad_b2
        self.W1 += self.lr * grad_W1
        self.b1 += self.lr * grad_b1

    def train(self, X, D, epochs=1):
        """
        Обучение сети на одном примере X, D в течение нескольких эпох.
        """
        errors = []
        for epoch in range(epochs):
            # Прямой проход
            output = self.feedforward(X)
            # Расчет среднеквадратичной ошибки (MSE) для информации
            mse = np.mean(np.square(D - output))
            errors.append(mse)
            # Обратный проход и обновление весов
            self.backpropagate(X, D)
            if (epoch + 1) % (max(1, epochs // 10)) == 0 or epoch == epochs -1 or epochs < 10 :
                print(f"Эпоха {epoch+1}/{epochs}, MSE: {mse:.6f}")
        return errors


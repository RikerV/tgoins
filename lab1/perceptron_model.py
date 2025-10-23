import numpy as np

class Perceptron:
    def __init__(self, num_inputs, num_outputs, learning_rate=0.1):
        """
        Инициализация перцептрона
        
        Args:
            num_inputs: количество входных сигналов (размерность изображения)
            num_outputs: количество выходных сигналов (количество классов)
            learning_rate: коэффициент скорости обучения
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = learning_rate
        
        # Инициализация весов и смещений малыми случайными значениями
        # Добавляем +1 к num_inputs для веса смещения (x0 = 1)
        self.weights = np.random.uniform(-0.1, 0.1, (num_inputs + 1, num_outputs))
        
        # Для хранения общей ошибки и изменений весов
        self.total_error = 0
        self.delta_weights = np.zeros((num_inputs + 1, num_outputs))

    def _sigmoid(self, x):
        """
        Сигмоидальная функция активации
        """
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # clip для избежания переполнения

    def _sigmoid_derivative(self, x):
        """
        Производная сигмоидальной функции
        """
        return x * (1 - x)

    def predict(self, inputs):
        """
        Вычисление выходных сигналов сети
        
        Args:
            inputs: входной вектор (x1,...,xn)
            
        Returns:
            Выходной вектор (y1,...,yk)
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Неверное количество входов. Ожидалось {self.num_inputs}, получено {len(inputs)}")
        
        # Добавляем единичный вход для смещения (x0 = 1)
        inputs_with_bias = np.append(inputs, 1)
        
        # Вычисление взвешенной суммы для каждого нейрона
        net_j = np.dot(inputs_with_bias, self.weights)
        
        # Применение функции активации
        y_j = self._sigmoid(net_j)
        
        return y_j

    def train_epoch(self, X_train, y_train):
        """
        Обучение на одной эпохе (всех обучающих примерах)
        
        Args:
            X_train: матрица обучающих примеров
            y_train: матрица целевых значений
            
        Returns:
            total_error: общая ошибка на эпохе
        """
        # Сброс накопленных значений
        self.total_error = 0
        self.delta_weights = np.zeros((self.num_inputs + 1, self.num_outputs))
        
        # Перемешивание данных
        permutation = np.random.permutation(len(X_train))
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]
        
        # Обучение на каждом примере
        for i in range(len(X_shuffled)):
            inputs = X_shuffled[i]
            targets = y_shuffled[i]
            
            # Шаг 4: Подача вектора на входы перцептрона
            inputs_with_bias = np.append(inputs, 1)  # x0 = 1
            
            # Шаг 5: Вычисление выходных сигналов
            net_j = np.dot(inputs_with_bias, self.weights)
            outputs = self._sigmoid(net_j)
            
            # Шаг 6: Вычисление ошибки
            error = targets - outputs
            E = 0.5 * np.sum(error ** 2)
            self.total_error += E / len(X_train)
            
            # Шаг 7: Вычисление коррекции весов (накопление)
            delta = error * self._sigmoid_derivative(outputs)
            self.delta_weights += np.outer(inputs_with_bias, delta)
        
        # Шаг 9: Коррекция весов после эпохи
        self.weights += self.lr * self.delta_weights
        
        return self.total_error

    def train(self, X_train, y_train, epochs=100, error_threshold=0.01, verbose=True):
        """
        Полный процесс обучения перцептрона
        
        Args:
            X_train: обучающие данные
            y_train: целевые значения
            epochs: максимальное количество эпох
            error_threshold: пороговое значение ошибки для остановки
            verbose: вывод информации о процессе обучения
            
        Returns:
            errors: список ошибок по эпохам
        """
        errors = []
        
        if verbose:
            print("Начало обучения методом градиентного спуска...")
            print(f"Параметры: LR={self.lr}, Эпох={epochs}, Порог ошибки={error_threshold}")
        
        for epoch in range(epochs):
            # Обучение на одной эпохе
            epoch_error = self.train_epoch(X_train, y_train)
            errors.append(epoch_error)
            
            # Вывод информации о процессе обучения
            if verbose and (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Эпоха {epoch + 1}/{epochs}, Общая ошибка: {epoch_error:.6f}")
            
            # Критерии останова (Шаг 8)
            # 1) Ошибка ниже порога
            if epoch_error < error_threshold:
                if verbose:
                    print(f"Обучение завершено: достигнут порог ошибки ({error_threshold}) на эпохе {epoch + 1}")
                break
            
            # 2) Ошибка меняется незначительно (после 10 эпох)
            if epoch > 10 and abs(errors[-1] - errors[-2]) < 1e-6:
                if verbose:
                    print(f"Обучение завершено: ошибка стабилизировалась на эпохе {epoch + 1}")
                break
        
        if verbose:
            if epoch == epochs - 1:
                print(f"Обучение завершено: достигнуто максимальное количество эпох ({epochs})")
            print(f"Финальная ошибка: {errors[-1]:.6f}")
        
        return errors

    def calculate_accuracy(self, X, y):
        """
        Вычисление точности классификации
        
        Args:
            X: входные данные
            y: целевые значения (one-hot encoding)
            
        Returns:
            accuracy: точность классификации
        """
        correct = 0
        for i in range(len(X)):
            prediction = self.predict(X[i])
            # Преобразуем непрерывные выходы в one-hot encoding
            pred_one_hot = np.zeros_like(prediction)
            pred_one_hot[np.argmax(prediction)] = 1
            
            if np.array_equal(pred_one_hot, y[i]):
                correct += 1
        
        return correct / len(X)

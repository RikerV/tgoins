import numpy as np
from data_preparation import load_symbol_data
from perceptron_model import Perceptron
import matplotlib.pyplot as plt

def calculate_accuracy(y_true, y_pred):
    """
    Вычисление точности классификации для one-hot encoded меток
    """
    correct_predictions = 0
    for i in range(len(y_true)):
        # Для непрерывных выходов находим индекс максимального значения
        true_class = np.argmax(y_true[i])
        pred_class = np.argmax(y_pred[i])
        if true_class == pred_class:
            correct_predictions += 1
    return correct_predictions / len(y_true)

def convert_to_one_hot(predictions):
    """
    Преобразование непрерывных выходов в one-hot encoding
    """
    one_hot = np.zeros_like(predictions)
    for i in range(len(predictions)):
        max_idx = np.argmax(predictions[i])
        one_hot[i, max_idx] = 1
    return one_hot

def main():
    print("Загрузка данных...")
    X_train, y_train, X_test, y_test = load_symbol_data()

    if X_train.size == 0 or X_test.size == 0:
        print("Ошибка при загрузке данных. Завершение работы.")
        return

    print(f"Обучающая выборка: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"Тестовая выборка: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    num_inputs = X_train.shape[1]
    num_outputs = y_train.shape[1]
    learning_rate = 0.1  # Увеличил learning rate для лучшей сходимости
    epochs = 100
    error_threshold = 0.01  # Порог ошибки для остановки

    # Инициализация перцептрона с градиентным спуском
    perceptron = Perceptron(num_inputs=num_inputs, num_outputs=num_outputs, learning_rate=learning_rate)
    print(f"\nМодель перцептрона инициализирована с {num_inputs} входами, {num_outputs} выходами.")
    print(f"Скорость обучения: {learning_rate}, Максимальное количество эпох: {epochs}")
    print(f"Порог ошибки: {error_threshold}\n")

    # Обучение модели (используем встроенный метод train)
    print("Начало обучения методом градиентного спуска...")
    errors = perceptron.train(X_train, y_train, epochs=epochs, error_threshold=error_threshold, verbose=True)

    print("\nОбучение завершено.\n")

    # Тестирование модели на тестовой выборке
    print("Тестирование модели на тестовой выборке...")
    
    # Получаем предсказания (непрерывные выходы)
    test_predictions_continuous = np.array([perceptron.predict(x) for x in X_test])
    
    # Преобразуем в one-hot encoding для сравнения
    test_predictions_one_hot = convert_to_one_hot(test_predictions_continuous)

    char_map_inv = {tuple(v): k for k, v in {
        'λ': [1, 0, 0, 0], 'φ': [0, 1, 0, 0],
        'η': [0, 0, 1, 0], 'γ': [0, 0, 0, 1]}.items()
    }

    print("\nРезультаты на тестовой выборке:")
    for i in range(len(X_test)):
        true_label_one_hot = y_test[i]
        pred_label_one_hot = test_predictions_one_hot[i]
        pred_continuous = test_predictions_continuous[i]

        true_char = char_map_inv.get(tuple(true_label_one_hot), "Неизвестно")
        pred_char = char_map_inv.get(tuple(pred_label_one_hot), "Неизвестно (ошибка классификации)")

        correct_str = "Верно" if np.array_equal(true_label_one_hot, pred_label_one_hot) else "Ошибка"
        
        # Выводим также непрерывные значения для информации
        print(f"Образец {i+1}: Истина: {true_char}")
        print(f"              Предсказано: {pred_char} {pred_continuous} -> {correct_str}")

    # Расчет точности
    test_accuracy = calculate_accuracy(y_test, test_predictions_one_hot)
    print(f"\nИтоговая точность на тестовой выборке: {test_accuracy:.4f} ({int(test_accuracy*len(y_test))}/{len(y_test)})")

    # Построение графика ОШИБКИ (а не количества ошибок классификации)
    if errors:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='red')
        plt.title('Общая ошибка (MSE) по эпохам - Градиентный спуск')
        plt.xlabel('Эпоха')
        plt.ylabel('Среднеквадратичная ошибка (MSE)')
        plt.grid(True)
        plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации
        plt.show()

    # Демонстрация предсказания для одного тестового образца
    print("\nДемонстрация предсказания для одного тестового образца:")
    if len(X_test) > 0:
        sample_idx = 0
        single_test_input = X_test[sample_idx]
        single_test_target = y_test[sample_idx]

        prediction_single = perceptron.predict(single_test_input)
        prediction_one_hot = convert_to_one_hot([prediction_single])[0]
        
        true_char_single = char_map_inv.get(tuple(single_test_target), "Неизвестно")
        pred_char_single = char_map_inv.get(tuple(prediction_one_hot), "Неизвестно")

        print(f"Входной тестовый образец {sample_idx+1} (Истинный символ: {true_char_single})")
        print(f"Предсказанный символ: {pred_char_single}")
        print(f"Выходные значения: {prediction_single}")
        print(f"Интерпретация: символ '{pred_char_single}' (нейрон {np.argmax(prediction_single)})")

        IMAGE_SIZE = (24, 24)
        plt.imshow(single_test_input.reshape(IMAGE_SIZE), cmap='gray')
        plt.title(f"Тестовый образец\nИстина: {true_char_single}, Предсказано: {pred_char_single}")
        plt.colorbar()
        plt.show()

    # Дополнительная информация о работе сети
    print("\nДополнительная информация:")
    print(f"Финальная ошибка обучения: {errors[-1]:.6f}")
    print(f"Количество пройденных эпох: {len(errors)}")

if __name__ == '__main__':
    main()

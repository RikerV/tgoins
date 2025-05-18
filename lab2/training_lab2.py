import numpy as np
from data_preparation import load_symbol_data # Убедись, что он в той же папке
from perceptron_reinforced_model import ReinforcedPerceptron
import matplotlib.pyplot as plt

def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        if np.array_equal(true, pred):
            correct_predictions += 1
    return correct_predictions / len(y_true)

def main():
    # 0. Фиксация seed для воспроизводимости (важно для сравнения с Лаб 1 и между запусками)
    # np.random.seed(42) # Используй тот же seed, что и в Лаб 1, если хочешь сравнить влияние алгоритма

    print("Загрузка данных...")
    X_train, y_train, X_test, y_test = load_symbol_data()

    if X_train.size == 0 or X_test.size == 0:
        print("Ошибка при загрузке данных. Завершение работы.")
        return

    print(f"Обучающая выборка: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"Тестовая выборка: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    # 2. Инициализация модели
    num_inputs = X_train.shape[1]
    num_outputs = y_train.shape[1]
    max_epochs = 500 # Максимальное количество эпох, чтобы избежать бесконечного цикла

    perceptron = ReinforcedPerceptron(num_inputs=num_inputs, num_outputs=num_outputs)
    print(f"\nМодель ReinforcedPerceptron инициализирована с {num_inputs} входами, {num_outputs} выходами.")
    print(f"Максимальное количество эпох: {max_epochs}\n")

    # 3. Обучение модели
    # Критерий останова: веса не меняются ИЛИ нет ошибок на всей обучающей выборке
    epoch_misclassifications = [] # Количество ошибок на обучающей выборке за эпоху

    print("Начало обучения...")
    for epoch in range(max_epochs):
        total_weights_changed_in_epoch = False # Флаг, изменились ли веса в этой эпохе
        num_misclassifications_current_epoch = 0

        # Перемешивание обучающих данных
        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(len(X_train_shuffled)):
            inputs = X_train_shuffled[i]
            targets = y_train_shuffled[i]

            # Перед обучением на примере, проверим, есть ли ошибка
            current_prediction = perceptron.predict(inputs)
            if not np.array_equal(current_prediction, targets):
                num_misclassifications_current_epoch += 1

            weights_changed_for_sample = perceptron.train_single_example(inputs, targets)
            if weights_changed_for_sample:
                total_weights_changed_in_epoch = True

        epoch_misclassifications.append(num_misclassifications_current_epoch)

        # Оценка точности на обучающей выборке (для контроля)
        train_predictions_epoch = np.array([perceptron.predict(x) for x in X_train])
        train_accuracy_epoch = calculate_accuracy(y_train, train_predictions_epoch)

        if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1 or num_misclassifications_current_epoch == 0:
            print(f"Эпоха {epoch + 1}/{max_epochs}, "
                  f"Ошибок на обуч. (до коррекции): {num_misclassifications_current_epoch}/{len(X_train)}, "
                  f"Точность на обуч. (после коррекции): {train_accuracy_epoch:.4f}, "
                  f"Веса изменены в эпохе: {total_weights_changed_in_epoch}")

        if num_misclassifications_current_epoch == 0 and not total_weights_changed_in_epoch : # Идеальный случай: нет ошибок и веса стабильны
             print(f"Обучение завершено на эпохе {epoch + 1}: нет ошибок на обучающей выборке и веса не изменялись.")
             break
        if train_accuracy_epoch == 1.0 and num_misclassifications_current_epoch == 0 : # Если после эпохи коррекции все правильно
            print(f"Обучение завершено на эпохе {epoch + 1}: достигнута 100% точность на обучающей выборке.")
            # Дополняем массив ошибок, если вышли раньше
            remaining_epochs = max_epochs - (epoch + 1)
            if remaining_epochs > 0 and epoch_misclassifications:
                 epoch_misclassifications.extend([0] * remaining_epochs)
            break


    if epoch == max_epochs - 1 and train_accuracy_epoch < 1.0:
        print(f"Обучение остановлено по достижении максимального числа эпох ({max_epochs}).")
    print("Обучение завершено.\n")

    # 4. Тестирование модели
    print("Тестирование модели на тестовой выборке...")
    test_predictions = np.array([perceptron.predict(x) for x in X_test])
    char_map_inv = {tuple(v): k for k, v in {
        'λ': [1, 0, 0, 0], 'φ': [0, 1, 0, 0],
        'η': [0, 0, 1, 0], 'γ': [0, 0, 0, 1]}.items()
    }

    print("\nРезультаты на тестовой выборке:")
    for i in range(len(X_test)):
        true_label_one_hot = y_test[i]
        pred_label_one_hot = test_predictions[i]
        true_char = char_map_inv.get(tuple(true_label_one_hot), "Неизвестно")
        pred_char = char_map_inv.get(tuple(pred_label_one_hot), "Неизвестно (ошибка классификации)")
        correct_str = "Верно" if np.array_equal(true_label_one_hot, pred_label_one_hot) else "Ошибка"
        print(f"Образец {i+1}: Истина: {true_char} ({true_label_one_hot}), Предсказано: {pred_char} ({pred_label_one_hot}) -> {correct_str}")

    test_accuracy = calculate_accuracy(y_test, test_predictions)
    print(f"\nИтоговая точность на тестовой выборке: {test_accuracy:.4f} ({int(test_accuracy*len(y_test))}/{len(y_test)})")

    # 5. Построение графика ошибки
    if epoch_misclassifications: # Проверяем, что список не пуст
        plt.figure(figsize=(10, 6))
        # Используем range(1, len(...) + 1) для нумерации эпох с 1
        epochs_ran = len(epoch_misclassifications)
        plt.plot(range(1, epochs_ran + 1), epoch_misclassifications[:epochs_ran], marker='o', linestyle='-')
        plt.title('Количество ошибок на обучающей выборке (до коррекции) по эпохам (Метод подкрепления)')
        plt.xlabel('Эпоха')
        plt.ylabel('Количество неправильно классифицированных образов')
        plt.grid(True)
        if epochs_ran > 0: # Добавляем метки только если были эпохи
            plt.xticks(range(1, epochs_ran + 1, max(1, epochs_ran // 10))) # Метки на оси X
        plt.show()

if __name__ == '__main__':
    main()
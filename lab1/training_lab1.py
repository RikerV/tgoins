import numpy as np
from data_preparation import load_symbol_data
from perceptron_model import Perceptron
import matplotlib.pyplot as plt # Для построения графика ошибки

def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        if np.array_equal(true, pred):
            correct_predictions += 1
    return correct_predictions / len(y_true)

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
    learning_rate = 0.01 # Можно поэкспериментировать с этим значением
    epochs = 100         # Количество эпох обучения

    perceptron = Perceptron(num_inputs=num_inputs, num_outputs=num_outputs, learning_rate=learning_rate)
    print(f"\nМодель персептрона инициализирована с {num_inputs} входами, {num_outputs} выходами.")
    print(f"Скорость обучения: {learning_rate}, Количество эпох: {epochs}\n")

    epoch_errors = [] # Для отслеживания общей ошибки на каждой эпохе

    print("Начало обучения...")
    for epoch in range(epochs):
        total_epoch_error = 0 
        num_misclassifications_epoch = 0

        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(len(X_train_shuffled)):
            inputs = X_train_shuffled[i]
            targets = y_train_shuffled[i]

            error_vector = perceptron.train_single_example(inputs, targets)
            total_epoch_error += np.sum(np.abs(error_vector))
            if not np.all(error_vector == 0): 
                num_misclassifications_epoch +=1


        epoch_errors.append(num_misclassifications_epoch) 

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1: 
            # Оценка точности на обучающей выборке (для контроля)
            train_predictions = np.array([perceptron.predict(x) for x in X_train])
            train_accuracy = calculate_accuracy(y_train, train_predictions)
            print(f"Эпоха {epoch + 1}/{epochs}, Ошибок на обуч. выборке: {num_misclassifications_epoch}/{len(X_train)}, Точность на обуч.: {train_accuracy:.4f}")

   
        if num_misclassifications_epoch == 0:
            print(f"Обучение завершено досрочно на эпохе {epoch + 1}: нет ошибок на обучающей выборке.")
            remaining_epochs = epochs - (epoch + 1)
            if remaining_epochs > 0:
                epoch_errors.extend([0] * remaining_epochs)
            break
    print("Обучение завершено.\n")

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

    if epoch_errors:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_errors) + 1), epoch_errors, marker='o', linestyle='-')
        plt.title('Количество ошибок на обучающей выборке по эпохам')
        plt.xlabel('Эпоха')
        plt.ylabel('Количество неправильно классифицированных образов')
        plt.grid(True)
        plt.xticks(range(1, len(epoch_errors) + 1)) # Чтобы метки были целыми числами эпох
        plt.show()

    print("\nДемонстрация предсказания для одного тестового образца:")
    if len(X_test) > 0:
        sample_idx = 0 # Берем первый тестовый образец
        single_test_input = X_test[sample_idx]
        single_test_target = y_test[sample_idx]

        prediction_single = perceptron.predict(single_test_input)
        true_char_single = char_map_inv.get(tuple(single_test_target), "Неизвестно")
        pred_char_single = char_map_inv.get(tuple(prediction_single), "Неизвестно")

        print(f"Входной тестовый образец {sample_idx+1} (Истинный символ: {true_char_single})")
        print(f"Предсказанный символ: {pred_char_single} ({prediction_single})")

        IMAGE_SIZE = (24, 24) # должно быть доступно или передано
        plt.imshow(single_test_input.reshape(IMAGE_SIZE), cmap='gray')
        plt.title(f"Тестовый образец. Истина: {true_char_single}, Предсказано: {pred_char_single}")
        plt.show()


if __name__ == '__main__':
    main()
import numpy as np
from neural_network_bpa import NeuralNetworkBPA
import matplotlib.pyplot as plt # Для графика

def main():
    input_dim = 2
    hidden_dim = 3
    output_dim = 3
    learning_rate_val = 0.25
    epochs_to_train = 100 # Количество эпох для демонстрации
    # np.random.seed(42) 

    nn = NeuralNetworkBPA(input_size=input_dim,
                           hidden_size=hidden_dim,
                           output_size=output_dim,
                           learning_rate=learning_rate_val)

    print("--- Начальное состояние сети ---")
    print("Начальные веса W1:\n", nn.W1)
    print("Начальные смещения b1:\n", nn.b1)
    print("Начальные веса W2:\n", nn.W2)
    print("Начальные смещения b2:\n", nn.b2)

    # Входной вектор X и целевой вектор D 
    X_sample = np.array([[0.4, 0.4]])
    D_sample = np.array([[0.3, 0.5, 0.8]])

    print(f"\nВход X: {X_sample}")
    print(f"Цель D: {D_sample}")

    # Выход сети до обучения
    initial_output = nn.feedforward(X_sample)
    initial_mse = np.mean(np.square(D_sample - initial_output))
    print(f"\nВыход до обучения: {initial_output}")
    print(f"MSE до обучения: {initial_mse:.6f}")
    print(f"\n--- Обучение на {epochs_to_train} эпох ---")
    # Обучаем основную сеть nn
    training_errors_mse = []
    for epoch in range(epochs_to_train):
        output = nn.feedforward(X_sample)
        mse = np.mean(np.square(D_sample - output))
        training_errors_mse.append(mse)
        
        nn.backpropagate(X_sample, D_sample) # Обновляем веса
        
        if (epoch + 1) % (max(1, epochs_to_train // 10)) == 0 or epoch == epochs_to_train -1 or epochs_to_train < 10 :
            print(f"Эпоха {epoch+1}/{epochs_to_train}, MSE: {mse:.6f}")


    print("\n--- Состояние сети после обучения ---")
    print("Финальные веса W1:\n", nn.W1)
    print("Финальные смещения b1:\n", nn.b1) # Можно раскомментировать, если нужно
    print("Финальные веса W2:\n", nn.W2)
    print("Финальные смещения b2:\n", nn.b2)

    final_output = nn.feedforward(X_sample) # Получаем финальный выход с обновленными весами
    final_mse = np.mean(np.square(D_sample - final_output)) # Это будет MSE *после* последнего обновления
    # Если нужно MSE *до* последнего обновления на последней эпохе, то это training_errors_mse[-1]
    print(f"\nФинальный выход после {epochs_to_train} эпох обучения: {final_output}")
    print(f"Финальный MSE (после {epochs_to_train} обновлений): {final_mse:.6f}")
    if training_errors_mse:
        print(f"MSE на последней эпохе (до обновления): {training_errors_mse[-1]:.6f}")


    # Построение графика ошибки
    if training_errors_mse:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_errors_mse) + 1), training_errors_mse, marker='.', linestyle='-')
        plt.title('Ошибка обучения (MSE) по эпохам')
        plt.xlabel('Эпоха')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.xticks(range(0, len(training_errors_mse) + 1, max(1, len(training_errors_mse)//10)))
        plt.show()

if __name__ == '__main__':
    main()
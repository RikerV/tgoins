from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def generate_char_image(character, font_path, image_size=(24, 24), font_size_ratio=0.8, threshold=128):
    #создание картинки символа
    try:
        font_size = int(image_size[1] * font_size_ratio)
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Ошибка: Не удалось загрузить шрифт: {font_path}") 
        return None
    image = Image.new('RGBA', image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        text_bbox = draw.textbbox((0, 0), character, font=font)
    except AttributeError:
        text_width, text_height = draw.textsize(character, font=font)
        text_bbox = (0, 0, text_width, text_height)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (image_size[0] - text_width) / 2 - text_bbox[0]
    y = (image_size[1] - text_height) / 2 - text_bbox[1]
    draw.text((x, y), character, font=font, fill=(0, 0, 0, 255))
    gray_image = image.convert('L')
    binary_image_array = np.array(gray_image)
    binary_image_processed = (binary_image_array < threshold).astype(int)
    return binary_image_processed.flatten()


def load_symbol_data():
    chars_to_generate = ['λ', 'φ', 'η', 'γ']
    char_labels = {
        'λ': [1, 0, 0, 0],
        'φ': [0, 1, 0, 0],
        'η': [0, 0, 1, 0],
        'γ': [0, 0, 0, 1],
    }
    base_font_path_win = "C:/Windows/Fonts/"

    font_files_train = [
        "arial.ttf",
        "times.ttf",
        "verdana.ttf",
        "calibri.ttf",
    ]
    font_file_test = "consola.ttf"

    font_paths_train = [os.path.join(base_font_path_win, f) for f in font_files_train]
    font_path_test = os.path.join(base_font_path_win, font_file_test)

    # Проверка существования шрифтов
    missing_fonts = []
    for fp in font_paths_train + [font_path_test]:
        if not os.path.exists(fp):
            missing_fonts.append(fp)
    if missing_fonts:
        print("Предупреждение: Следующие файлы шрифтов не найдены:")
        for mf in missing_fonts:
            print(f"- {mf}")
        print("Пожалуйста, проверьте пути к шрифтам или скопируйте их в папку проекта/fonts.")

    IMAGE_SIZE = (24, 24)
    THRESHOLD = 150

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print("Генерация обучающей выборки...") 
    for char_unicode in chars_to_generate:
        for font_p in font_paths_train:
            img_vector = generate_char_image(char_unicode, font_p, image_size=IMAGE_SIZE, threshold=THRESHOLD)
            if img_vector is not None:
                X_train_list.append(img_vector)
                y_train_list.append(char_labels[char_unicode])
            else:
                print(f"Не удалось сгенерировать '{char_unicode}' шрифтом {os.path.basename(font_p)}")


    print("\nГенерация тестовой выборки...") 
    for char_unicode in chars_to_generate:
        img_vector = generate_char_image(char_unicode, font_path_test, image_size=IMAGE_SIZE, threshold=THRESHOLD)
        if img_vector is not None:
            X_test_list.append(img_vector)
            y_test_list.append(char_labels[char_unicode])
        else:
            print(f"Не удалось сгенерировать '{char_unicode}' шрифтом {os.path.basename(font_path_test)}")


    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    # Проверка, что данные не пустые
    if X_train.size == 0 or X_test.size == 0:
        print("Ошибка: Обучающая или тестовая выборка пуста. Проверьте наличие шрифтов и процесс генерации.")


    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train_data, y_train_data, X_test_data, y_test_data = load_symbol_data()

    if X_train_data is not None and X_train_data.size > 0 :
        print(f"Размер обучающей выборки: X_train.shape = {X_train_data.shape}, y_train.shape = {y_train_data.shape}")
        print(f"Размер тестовой выборки: X_test.shape = {X_test_data.shape}, y_test.shape = {y_test_data.shape}")

        print("\nПример первого обучающего образа (первые 20 элементов):")
        print(X_train_data[0][:20])
        print("Метка первого обучающего образа:")
        print(y_train_data[0])

        import matplotlib.pyplot as plt
        from datetime import datetime
        if X_train_data.shape[1] == 24*24:
            plt.imshow(X_train_data[0].reshape((24, 24)), cmap='gray')
            label_key = None
            target_label_list = list(y_train_data[0])
            for char_k, char_v_list in {
                'λ': [1, 0, 0, 0], 'φ': [0, 1, 0, 0],
                'η': [0, 0, 1, 0], 'γ': [0, 0, 0, 1]
            }.items():
                if char_v_list == target_label_list:
                    label_key = char_k
                    break
            plt.title(f"Символ: {label_key if label_key else 'Неизвестно'}")
            plt.show()
    else:
        print("Данные не были загружены/сгенерированы.")
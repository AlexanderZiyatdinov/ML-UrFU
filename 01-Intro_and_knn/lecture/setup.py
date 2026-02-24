import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from IPython.display import display, clear_output
import time
import warnings
warnings.filterwarnings('ignore')


def get_grid(data, border=1., step=.05):  # получаем все точки плоскости
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step),
                       np.arange(y_min, y_max, step))


def plot_model(X_train, y_train, clf, title=None, proba=False):
    xx, yy = get_grid(X_train)  # получаем все точки плоскости
    plt.figure(figsize=(7, 7))
    # предсказываем значения для каждой точки плоскости

    if proba:  # нужно ли предсказывать вероятности
        predicted = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[
            :, 1].reshape(xx.shape)
    else:
        predicted = clf.predict(
            np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Отрисовка плоскости
    ax = plt.gca()
    ax.pcolormesh(xx, yy, predicted, cmap='spring')

    # Отрисовка точек

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
               s=100, cmap='spring', edgecolors='b')
    colors = ['purple', 'yellow', 'orange']
    patches = []
    for yi in np.unique(y_train):
        patches.append(mpatches.Patch(
            color=colors[int(yi)], label='$y_{pred}=$'+str(int(yi))))
    ax.legend(handles=patches)
    plt.title(title)
    return clf


def plot_reg(X, y, clf_dtc, X_test, kind='plot'):
    clf_dtc.fit(X, y)
    Y_test = clf_dtc.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, cmap='bwr', s=50, alpha=1)
    if kind == 'scatter':
        plt.scatter(X_test, Y_test, color='r', alpha=.25, s=15)
    elif kind == 'plot':
        plt.plot(X_test, Y_test, color='r', alpha=1)
    else:
        raise ValueError(f'Can\'t plot "{kind}"')
    plt.grid()


def runge_example():
    """
    Демонстрация переобучения на примере аппроксимации функции Рунге.
    Анимация показывает, как с увеличением степени полинома происходит переобучение.
    """
    # Функция Рунге
    def runge_function(x):
        return 1 / (1 + 25 * x**2)

    # Генерация данных
    np.random.seed(42)
    n_samples = 20
    X = np.linspace(-1, 1, n_samples)
    y_true = runge_function(X)
    y = y_true + np.random.normal(0, 0.05, n_samples)  # Добавляем шум

    # Используем только обучающую выборку
    X_train = X
    y_train = y

    # Для плавной визуализации
    x_plot = np.linspace(-1, 1, 500)
    y_plot_true = runge_function(x_plot)

    # Списки для хранения истории
    degrees = []
    train_errors = []
    overfitting_detected = False
    best_degree = 1
    min_train_error = float('inf')

    # Основной цикл анимации
    for degree in range(1, 41):
        # Очищаем вывод для анимации
        clear_output(wait=True)

        # Создаем новую фигуру для каждого кадра
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Переобучение при аппроксимации функции Рунге (Степень: {degree})',
                     fontsize=16, fontweight='bold')

        # Настройка первого графика
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Настройка второго графика
        ax2.set_xlabel('Степень полинома', fontsize=12)
        ax2.set_ylabel('Ошибка (MSE)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 41)
        ax2.set_ylim(0, 0.1)

        # Исходные данные (только обучающая выборка)
        ax1.scatter(X_train, y_train, color='blue', s=60, label='Обучающая выборка',
                    zorder=5, alpha=0.7)
        ax1.plot(x_plot, y_plot_true, 'k--', label='Истинная функция',
                 alpha=0.7, linewidth=2.5)

        # Создаем и обучаем модель
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train.reshape(-1, 1), y_train)

        # Предсказания
        y_pred_plot = model.predict(x_plot.reshape(-1, 1))
        y_pred_train = model.predict(X_train.reshape(-1, 1))

        # Вычисление ошибок (только на обучающей выборке)
        train_error = np.mean((y_train - y_pred_train) ** 2)

        # Сохраняем ошибки
        degrees.append(degree)
        train_errors.append(train_error)

        # Обновляем лучшую степень (минимальная ошибка на обучении)
        if train_error < min_train_error:
            min_train_error = train_error
            best_degree = degree

        # Рисуем аппроксимацию
        ax1.plot(x_plot, y_pred_plot, 'b-', label='Аппроксимация',
                 linewidth=2, alpha=0.9)

        # Рисуем кривую обучения (только ошибка на обучении)
        ax2.plot(degrees, train_errors, 'b-', label='Ошибка на обучении',
                 linewidth=2.5, marker='o', markersize=5)

        # Проверка на переобучение
        title_color = 'black'
        warning_text = ''

        if len(train_errors) >= 3:
            # Проверяем, когда ошибка на обучении становится очень маленькой
            # и полином начинает сильно колебаться
            if train_error < 0.001 and degree > 20:
                if not overfitting_detected:
                    overfitting_detected = True
                    ax1.set_facecolor((1, 0.9, 0.9))
                    title_color = 'red'
                    warning_text = 'СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ!'

                    # Добавляем вертикальную линию на втором графике
                    ax2.axvline(x=degree, color='red', linestyle=':', alpha=0.7,
                                linewidth=2, label='Начало переобучения')

            # Альтернативный критерий: когда полином имеет слишком много экстремумов
            # Считаем количество пересечений с истинной функцией
            y_pred_smooth = model.predict(x_plot.reshape(-1, 1))
            diff_sign = np.diff(np.sign(y_pred_smooth - y_plot_true))
            num_crossings = np.sum(diff_sign != 0)

            if num_crossings > degree and degree > 15:
                if not overfitting_detected:
                    overfitting_detected = True
                    ax1.set_facecolor((1, 0.95, 0.9))
                    title_color = 'orange'
                    warning_text = 'ПЕРЕОБУЧЕНИЕ (много колебаний)'

                    # Добавляем вертикальную линию на втором графике
                    ax2.axvline(x=degree, color='orange', linestyle=':', alpha=0.7,
                                linewidth=2, label='Начало переобучения')

        # Обновляем заголовок с цветом
        ax1.set_title(f'Аппроксимация полиномом {degree}-й степени',
                      fontsize=14, fontweight='bold', color=title_color)
        ax2.set_title('Кривая обучения', fontsize=14, fontweight='bold')

        # Легенды
        ax1.legend(loc='upper right', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        # Показываем график
        display(fig)

        # Вывод информации в консоль
        print(f"Шаг {degree}/40: Степень {degree}, Ошибка = {train_error:.6f}")
        if warning_text:
            print(f"   {warning_text}")

        # Пауза для анимации (1 секунда)
        time.sleep(1.0)

        # Закрываем фигуру, чтобы не накапливать в памяти
        plt.close(fig)

    # Первый график - лучшая аппроксимация
    model_best = make_pipeline(
        PolynomialFeatures(best_degree), LinearRegression())
    model_best.fit(X_train.reshape(-1, 1), y_train)
    y_pred_best = model_best.predict(x_plot.reshape(-1, 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    runge_example()

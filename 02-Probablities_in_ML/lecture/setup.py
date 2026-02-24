import numpy as np
import ipywidgets as widgets

def monty_hall_simulation(n_trials=1000):
    """
    Симуляция парадокса Монти Холла
    """
    np.random.seed(42)

    stay_wins = 0
    switch_wins = 0

    for _ in range(n_trials):
        # Приз случайно размещен
        prize_door = np.random.randint(0, 3)
        # Игрок случайно выбирает
        choice = np.random.randint(0, 3)

        # Если не меняем выбор
        if choice == prize_door:
            stay_wins += 1

        # Если меняем выбор
        if choice != prize_door:
            switch_wins += 1

    return stay_wins / n_trials, switch_wins / n_trials


def run_monty_hall(n_trials=1000):
    stay_prob, switch_prob = monty_hall_simulation(n_trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ['Не менять', 'Менять']
    probs = [stay_prob, switch_prob]
    colors = ['#ff6b6b', '#4ecdc4']

    bars = ax.bar(strategies, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Вероятность выигрыша')
    ax.set_title(f'Парадокс Монти Холла (симуляция {n_trials} игр)')

    # Добавляем теоретические значения
    ax.axhline(y=1/3, color='red', linestyle='--', alpha=0.5, label='Теория: 1/3')
    ax.axhline(y=2/3, color='blue', linestyle='--', alpha=0.5, label='Теория: 2/3')

    # Добавляем значения на столбцы
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n💡 Вывод: Меняя выбор, вы выигрываете в {switch_prob/stay_prob:.1f} раза чаще!")

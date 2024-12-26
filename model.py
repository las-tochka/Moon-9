import numpy as np
import matplotlib.pyplot as plt
# Задаем параметр (измените значения в зависимости от вашего контекста)

phi = np.degrees(90)
phi_final1 = 60
phi_final2 = 45
phi_final3 = 30
U = 77.65
S = 83.28
m0 = 292000
gamma1 = 1861
gamma2 = 466
gamma3 = 317
p = 1.2255
M = 5.29 * 10 ** 22
G = 6.67430 * 10 ** -11
g = 10
C = 0.5
R = 8.314
T = 273
R_planet = 600_000
# Начальные условия
h = [70]  # Начальная высота
v = 50  # Начальная скорость
dt = 1  # Шаг времени
t_max = 350  # Максимальное время


# Функции для первого этапа (t < t1)a
def ax_stage1(phi, t, m0, v, h):
    air_density = p * np.exp(-g * M * h / R_planet)
    return (U * t * np.cos(np.cos(phi)) - C * S * air_density * (v ** 2) / 2 * np.cos(np.cos(phi))) / m0


def ay_stage1(phi, t, m0, v, h):
    air_density = p * np.exp(-g * h * M / R_planet)
    return (U * t * np.sin(np.sin(phi)) - C * S * air_density * (v ** 2) / 2 * np.sin(np.sin(phi))) / m0 - (G * M) / (h + R_planet) ** 2



time = np.arange(0, t_max, dt)
heights = []


# Основной расчет
for t in time:
    print(str(h[-1]))
    if t < 72:
        gamma = gamma1 + gamma2
    if t == 73:
        m0 -= 48000
    if t == 120:
        m0 -= 63000
    elif 72 <= t < 120:
        U = 64.2
        gamma = gamma2
    elif 120 <= t < 290:
        U = 0
        gamma = 0
    else:
        U = 50.2
        gamma = gamma3

    m0 -= gamma * dt  # Уменьшаем массу

    if t == 86:
        phi = np.degrees(90 + (phi_final1 - 90) * (t / 8))
    if t == 94:
        phi = np.degrees(60 + (phi_final2 - 60) * (t / 9))
    if t == 295:
        phi = np.degrees(0)

    # Вычисляем ускорения
    a_x = ax_stage1(phi, t, m0, v, h[-1])
    a_y = ay_stage1(phi, t, m0, v, h[-1])

    # Обновляем скорость и высоту
    v_x = v * np.cos(phi) + a_x * dt
    v_y = v * np.sin(phi) + a_y * dt
    v = np.sqrt(v_x ** 2 + v_y ** 2)  # Полная скорость
    h_new = h[-1] + v_y * dt
    h.append(h_new)

    # Записываем текущую высоту
    heights.append(h_new)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(time, heights, label='Траектория высоты')
plt.title('Зависимость высоты от времени', fontsize=16)
plt.xlabel('Время (с)', fontsize=14)
plt.ylabel('Высота над поверхностью (м)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

plt.tight_layout()
plt.show()

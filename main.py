import numpy as np
import matplotlib.pyplot as plt
EPSILON = 0.001


def str_arr(arr):
    s = ''
    for el in arr:
        s += '{:.5f}'.format(el).ljust(9)
    return s + '\n'


def tau(t, a, n):
    return (t - a) / n


def delta(a, b):
    return [abs(d) for d in [x - y for x, y in zip(a, b)]]


def ti(i, t, a, n):
    return i * tau(t, a, n) + a


def diff_eq(x):
    return -(1 / (np.exp(x) - 2 * np.exp((x*x) / 2)))


def func(x, u):
    return (((1 - x) * np.exp(x) * np.power(u, 2)) - x * u)


def euler(x, t, a, n):
    u = [diff_eq(x[0])]
    for i in range(1, n + 1):
        u.append(u[i - 1] + tau(t, a, n) * func(x[i - 1], u[i - 1]))
    return u


def euler_sup(x, u, t, a, n):
    u_sup = [diff_eq(x[0])]
    for i in range(1, n + 1):
        u_sup.append(u[i - 1] + 2 * tau(t, a, n) * func(x[i - 1], u[i - 1]))
    return u_sup


def rk(x, t, a, n):
    u = [diff_eq(x[0])]
    for i in range(1, n+1):
        k1 = tau(t, a, n) * func(x[i - 1], u[i - 1])
        k2 = tau(t, a, n) * func(x[i - 1] + tau(t, a, n) / 2, u[i-1] + k1 / 2)
        k3 = tau(t, a, n) * func(x[i - 1] + tau(t, a, n) / 2, u[i-1] + k2 / 2)
        k4 = tau(t, a, n) * func(x[i - 1] + tau(t, a, n), u[i-1] + k3)
        u.append(u[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return u


def rk_sup(x, u, t, a, n):
    u_sup = [diff_eq(x[0])]
    for i in range(1, n+1):
        k1 = tau(t, a, n) * func(x[i - 1], u[i - 1])
        k2 = tau(t, a, n) * func(x[i - 1] + tau(t, a, n) / 2, u[i-1] + k1 / 2)
        k3 = tau(t, a, n) * func(x[i - 1] + tau(t, a, n) / 2, u[i-1] + k2 / 2)
        k4 = tau(t, a, n) * func(x[i - 1] + tau(t, a, n), u[i-1] + k3)
        u_sup.append(u[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 3)
    return u_sup


def main():
    a = 0
    n = 2
    t = 1.6
    u = []
    u_sup = []
    x = np.linspace(a, t, 1000)

    figs = []
    ax = []
    figs.append(plt.figure('Все графики'))
    figs.append(plt.figure("Коши"))
    figs.append(plt.figure('Эйлера'))
    figs.append(plt.figure('Рунге-Кутта'))
    for i in range(4):
        ax.append(figs[i].add_subplot())
        ax[i].grid()
        ax[i].set(xlabel='t', ylabel='u(t)')
    ax[1].plot(x, diff_eq(x), label='U(t)')
    ax[0].plot(x, diff_eq(x), linewidth=3, label='U(t)')

    print('Метод Эйлера')
    with open('euler.txt', 'w', encoding='utf8') as f:
        while True:
            x = [ti(i, t, a, n) for i in range(n + 1)]
            u = euler(x, t, a, n)
            u_sup = euler_sup(x, u, t, a, n)
            f.write(f'N = {n}\n')
            print(f'\rN = {n} 	\u03C4 = {tau(t, a, n):.7f}', end='\r')
            f.write(f'\u03C4 = {tau(t, a, n):.7f}\n')
            f.write(f'max \u0394 = {max(delta(u, u_sup)) / 15:.7f}\n')
            f.write('Приближенные значения:\n')
            f.write(str_arr(u)+'\n')
            if max(delta(u, u_sup)) < EPSILON:
                break
            n *= 2
    print(f'\rN = {n} 	\u03C4 = {tau(t, a, n):.7f}')
    print(f'max \u0394 = {max(delta(u, u_sup)):.7f}')
    print('Приближенные значения: ')
    print(str_arr(u), end='')
    print()
    print(f'Результат сравнения: {max(delta(u, diff_eq(np.array(x)))):.7f}')
    i = delta(u, diff_eq(np.array(x))).index(
        max(delta(u, diff_eq(np.array(x)))))
    print(f'tn = {x[i]:.1f} n = {i}')
    print(f'U({x[i]:.1f}) = {diff_eq(np.array(x))[i]:.7f}')
    print(f'u{i} = {u[i]:.7f}')
    ax[2].plot(x, u, label='Эйлера')
    ax[0].plot(x, u, '-', label='Эйлера')

    print('\nМетод Рунге-Кутта')
    n = 1
    with open('runge_kutta.txt', 'w', encoding='utf8') as f:
        while True:
            x = [ti(i, t, a, n) for i in range(n + 1)]
            u = rk(x, t, a, n)
            u_sup = rk_sup(x, u, t, a, n)
            f.write(f'N = {n}\n')
            print(f'\rN = {n} 	\u03C4 = {tau(t, a, n):.7f}', end='\r')
            f.write(f'\u03C4 = {tau(t, a, n):.7f}\n')
            f.write(f'max \u0394 = {max(delta(u, u_sup)) / 15:.7f}\n')
            f.write('Приближенные значения:\n')
            f.write(str_arr(u)+'\n')
            if max(delta(u, u_sup)) / 15 < EPSILON:
                break
            n *= 2
    print(f'\rN = {n} 	\u03C4 = {tau(t, a, n):.7f}')
    print('Приближенные значения: ')
    print(str_arr(u), end='')
    print(f'max \u0394 = {max(delta(u, u_sup)) / 15:.7f}')
    print(f'Результат сравнения: {max(delta(u, diff_eq(np.array(x)))):.13f}')
    i = delta(u, diff_eq(np.array(x))).index(
        max(delta(u, diff_eq(np.array(x)))))
    print(f'tn = {x[i]:.1f}    n = {i}')
    print(f'U({x[i]:.1f}) = {diff_eq(np.array(x))[i]:.7f}')
    print(f'u{i} = {u[i]:.7f}')
    ax[3].plot(x, u, label='Рунге-Кутта')
    ax[0].plot(x, u, ':', label='Рунге-Кутта')

    ax[0].legend()
    figs[0].savefig('all.png')
    figs[1].savefig('couchy.png')
    figs[2].savefig('euler.png')
    figs[3].savefig('rk.png')


if __name__ == "__main__":
    main()

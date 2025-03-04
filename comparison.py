from find_best_approximation import find_best_approximation
from least_square_approximation import least_square_approximation
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp

# 设置高精度计算
mp.dps = 40  # 设置 15 位小数精度

def comparison(data, degree):
    result_poly_a = find_best_approximation(data, degree)
    result_poly_b = least_square_approximation(data, degree)

    data = sorted(data, key=lambda item: item[0])
    x = np.array([point[0] for point in data])
    y = np.array([point[1] for point in data])
    x_poly = np.linspace(x[0], x[-1], 100)
    y_poly_a = result_poly_a(x_poly)
    y_poly_b = result_poly_b(x_poly)

    # 绘图
    plt.scatter(x, y, label='Original data')
    plt.plot(x_poly, y_poly_a, label='Approximation by new method')
    plt.plot(x_poly, y_poly_b, label='Approximation by least-squares')

    plt.legend()
    plt.title('Comparison: Least-squares vs. New Method')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

    # 使用 mpmath 进行高精度误差计算
    errors_a = [mp.mpf(abs(result_poly_a(point[0]) - point[1])) for point in data]
    errors_b = [mp.mpf(abs(result_poly_b(point[0]) - point[1])) for point in data]

    # 计算 MSE（均方误差）
    mse_a = mp.mpf(mp.fsum([e ** 2 for e in errors_a]) / len(errors_a))
    mse_b = mp.mpf(mp.fsum([e ** 2 for e in errors_b]) / len(errors_b))

    # 计算 MAE（平均绝对误差）
    mae_a = mp.mpf(mp.fsum(errors_a) / len(errors_a))
    mae_b = mp.mpf(mp.fsum(errors_b) / len(errors_b))

    # 计算 MAE_max（最大绝对误差）
    mae_max_a = mp.mpf(max(errors_a))
    mae_max_b = mp.mpf(max(errors_b))

    # 打印误差对比，保证高精度显示
    print('--- High Precision Error Comparison ---')
    print(f'New Method - MSE: {mp.nstr(mse_a, 40)}, MAE: {mp.nstr(mae_a, 40)}, MAE_max: {mp.nstr(mae_max_a, 40)}')
    print(f'Least Squares - MSE: {mp.nstr(mse_b, 40)}, MAE: {mp.nstr(mae_b, 40)}, MAE_max: {mp.nstr(mae_max_b, 40)}')

    return {
        "MSE": (mse_a, mse_b),
        "MAE": (mae_a, mae_b),
        "MAE_max": (mae_max_a, mae_max_b)
    }
import numpy as np

def least_square_approximation(data, max_degree):

    data = sorted(data, key=lambda item: item[0])
    x = np.array([point[0] for point in data])
    y = np.array([point[1] for point in data])

    best_poly = None
    min_error = float('inf')

    for deg in range(min(len(x)-1, max_degree) + 1):  # 确保阶数不超过数据点数 - 1
        coefficients = np.polyfit(x, y, deg)
        polynomial = np.poly1d(coefficients)

        # 计算均方误差 (MSE)
        y_pred = polynomial(x)
        error = np.mean((y - y_pred) ** 2)

        # 选择误差最小的多项式
        if error < min_error:
            min_error = error
            best_poly = polynomial

    return best_poly
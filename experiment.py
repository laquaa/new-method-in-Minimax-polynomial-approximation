import numpy as np
import pandas as pd
from comparison import comparison

np.random.seed(10)  # 保持结果可复现

# 选择不同数量的点，确保 -10 和 10 存在
num_points_list = [5, 7, 9, 11]

# 定义拟合的目标函数（最高 10 次）
functions = [
    {'func': lambda x: x + 1, 'label': 'Linear'},
    {'func': lambda x: x ** 2 + x + 1, 'label': 'Quadratic'},
    {'func': lambda x: x ** 3 + x ** 2 + x + 1, 'label': 'Cubic'},
    {'func': lambda x: x ** 4 + x ** 3 + x ** 2 + x + 1, 'label': 'Quartic'},
    {'func': lambda x: x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, 'label': 'Quintic'},
    {'func': lambda x: x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, 'label': 'Sextic'},
    {'func': lambda x: x ** 7 + x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, 'label': 'Septic'},
    {'func': lambda x: x ** 8 + x ** 7 + x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, 'label': 'Octic'},
    {'func': lambda x: x ** 9 + x ** 8 + x ** 7 + x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, 'label': 'Nonic'},
    {'func': lambda x: x ** 10 + x ** 9 + x ** 8 + x ** 7 + x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1,
     'label': 'Decic'}
]


# 生成不同分布方式的数据点，确保 -10 和 10 存在
def generate_points(n, distribution_type):
    if distribution_type == 'uniform':
        return np.linspace(-10, 10, n)

    elif distribution_type == 'middle-clustered':  # 中间点密集，边界点固定
        mid_n = (n - 2)  # 去掉 -10 和 10，剩余点数
        return np.concatenate((
            np.array([-10]),  # 确保 -10 存在
            np.linspace(-5, 5, mid_n),  # 中间区域更密集
            np.array([10])  # 确保 10 存在
        ))

    elif distribution_type == 'edge-clustered':  # 两端点密集，边界点固定
        edge_n = (n - 2) // 2  # 每侧分配点数
        mid_n = (n - 2) - 2 * edge_n  # 计算中间区域点数，保证 n 点总数正确
        return np.concatenate((
            np.array([-10]),  # 确保 -10 存在
            np.linspace(-8, -6, edge_n),  # 左端较密
            np.linspace(-5, 5, mid_n),  # 中间较稀疏
            np.linspace(6, 8, edge_n),  # 右端较密
            np.array([10])  # 确保 10 存在
        ))


# 创建 DataFrame 用于存储实验结果
columns = ["Function", "Distribution", "N", "MSE_New", "MSE_LSM", "MAE_New", "MAE_LSM", "MAE_Max_New", "MAE_Max_LSM"]
results_df = pd.DataFrame(columns=columns)

# 运行实验
for n_points in num_points_list:
    for func_info in functions:
        for distribution in ['uniform', 'middle-clustered', 'edge-clustered']:
            x_points = generate_points(n_points, distribution)
            data = [[x, func_info['func'](x)] for x in x_points]

            print(f"Comparison for {func_info['label']} ({distribution}, N={n_points}):")
            errors = comparison(data, 3)  # 方法的 degree 设为 3

            # 存入 DataFrame，保留 3 位有效数字
            new_row = pd.DataFrame([{
                "Function": func_info['label'],
                "Distribution": distribution,
                "N": n_points,
                "MSE_New": round(errors["MSE"][0], 3),
                "MSE_LSM": round(errors["MSE"][1], 3),
                "MAE_New": round(errors["MAE"][0], 3),
                "MAE_LSM": round(errors["MAE"][1], 3),
                "MAE_Max_New": round(errors["MAE_max"][0], 3),
                "MAE_Max_LSM": round(errors["MAE_max"][1], 3)
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

# 保存 DataFrame 到 CSV 文件
results_df.to_csv("experiment_results.csv", index=False)

print("\nAll results saved to 'experiment_results.csv'")

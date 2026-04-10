import numpy as np

def rate_3alpha(T: float) -> float:
    """3-alpha 反应率 q(T) 的温度相关部分"""
    T8 = T / 1.0e8
    # 防止数值溢出（T 过小时指数极大，但物理上 T 通常 >= 1e7 K）
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)

def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    """前向差分近似 dq/dT，使用相对步长 h = ΔT / T0"""
    if T0 <= 0:
        raise ValueError("温度必须为正数")
    delta_T = h * T0          # 关键：相对步长转为绝对步长
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + delta_T)
    return (q1 - q0) / delta_T

def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    """计算温度敏感性指数 nu = (T/q) * dq/dT"""
    q = rate_3alpha(T0)
    if q <= 0:
        return 0.0            # 避免除零，实际中不会发生
    dq_dT = finite_diff_dq_dT(T0, h)
    return (T0 / q) * dq_dT

def nu_table(T_values, h: float = 1e-8):
    """返回列表 [(T, nu(T)), ...]"""
    return [(T, sensitivity_nu(T, h)) for T in T_values]

# ========== 主程序（按要求输出） ==========
if __name__ == "__main__":
    temperatures = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    results = nu_table(temperatures, h=1e-8)

    for T, nu in results:
        print(f"{T:.3e} K : nu = {nu:.2f}")

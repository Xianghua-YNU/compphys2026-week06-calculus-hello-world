import math

def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    """复合梯形积分公式"""
    if n <= 0:
        raise ValueError("区间数 n 必须大于 0")
    h = (b - a) / n
    x = a
    s = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, n):
        x += h
        s += f(x)
    return s * h

def simpson_composite(f, a: float, b: float, n: int) -> float:
    """复合辛普森积分公式，要求 n 为偶数"""
    if n <= 0:
        raise ValueError("区间数 n 必须大于 0")
    if n % 2 != 0:
        raise ValueError("复合辛普森积分要求 n 为偶数")
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4.0 * f(a + i * h)
    for i in range(2, n-1, 2):
        s += 2.0 * f(a + i * h)
    return s * h / 3.0

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    """
    计算德拜积分 I(theta_d/T)，积分区间理论上为 [0, ∞)。
    由于被积函数衰减极快，采用足够大的上限（如 50）代替无穷大。
    """
    if T <= 0:
        raise ValueError("温度 T 必须大于 0")
    upper_limit = theta_d / T
    # 积分变量为 x，实际积分公式为：∫_0^{θ_D/T} x^4 e^x / (e^x - 1)^2 dx
    # 因此直接对 debye_integrand 在 [0, upper_limit] 上积分即可。
    # 注意：真正的德拜热容公式中积分上限是 θ_D/T，不需要扩展到无穷。
    # 这里的 debye_integrand 已经包含了被积函数。
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, upper_limit, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, upper_limit, n)
    else:
        raise ValueError("method 参数必须为 'trapezoid' 或 'simpson'")

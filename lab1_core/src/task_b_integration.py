import math

def debye_integrand(x: float) -> float:
    """德拜热容被积函数 f(x) = x^4 * e^x / (e^x - 1)^2"""
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    """复合梯形积分公式"""
    if n <= 0:
        raise ValueError("区间数 n 必须大于 0")
    h = (b - a) / n
    s = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, n):
        s += f(a + i * h)
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
    for i in range(2, n - 1, 2):
        s += 2.0 * f(a + i * h)
    return s * h / 3.0

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    """计算德拜积分 I(theta_d/T)"""
    if T <= 0:
        raise ValueError("温度 T 必须大于 0")
    upper_limit = theta_d / T

    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, upper_limit, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, upper_limit, n)
    else:
        raise ValueError("method 参数必须为 'trapezoid' 或 'simpson'")

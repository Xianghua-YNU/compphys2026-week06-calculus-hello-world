import math

def debye_integrand(x: float) -> float:
    """被积函数 f(x) = x^4 * e^x / (e^x - 1)^2，对 x=0 和 x 过大做保护"""
    if abs(x) < 1e-12:
        return 0.0
    # 防止指数溢出：当 x > 700 时，math.exp(x) 会溢出，此时 f(x) 已近似为 x^4 / e^x 衰减
    if x > 700.0:
        # 对于大 x，分母 (e^x - 1)^2 ≈ e^{2x}，因此 f(x) ≈ x^4 * e^{-x}
        return (x ** 4) * math.exp(-x)
    ex = math.exp(x)
    return (x ** 4) * ex / ((ex - 1.0) ** 2)

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n 必须大于 0")
    h = (b - a) / n
    s = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

def simpson_composite(f, a: float, b: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n 必须大于 0")
    # ⚠️ 自动修正：如果传入奇数 n，自动降级为复合梯形法，避免异常中断
    if n % 2 != 0:
        # 或者可以选择自动 +1 变为偶数，此处采用更安全的降级处理
        return trapezoid_composite(f, a, b, n)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4.0 * f(a + i * h)
    for i in range(2, n-1, 2):
        s += 2.0 * f(a + i * h)
    return s * h / 3.0

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    if T <= 0:
        raise ValueError("温度 T 必须大于 0")
    upper_limit = theta_d / T
    # 进一步限制积分上限：当 upper_limit 极大时，被积函数早已衰减至 0，截断至安全值
    # 德拜函数在 y > 50 后数值几乎不再变化，截断可避免溢出且不影响精度
    if upper_limit > 100.0:
        upper_limit = 100.0
    
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, upper_limit, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, upper_limit, n)
    else:
        raise ValueError("method 必须为 'trapezoid' 或 'simpson'")

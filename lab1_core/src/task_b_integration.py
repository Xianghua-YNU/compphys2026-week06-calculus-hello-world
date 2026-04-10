import numpy as np

def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = np.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be > 0")
    x = np.linspace(a, b, n+1)
    y = np.vectorize(f)(x)
    return np.trapz(y, x)

def simpson_composite(f, a: float, b: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be > 0")
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    x = np.linspace(a, b, n+1)
    y = np.vectorize(f)(x)
    return (b-a)/(3*n) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    if T <= 0:
        raise ValueError("T must be > 0")
    upper_limit = theta_d / T
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, upper_limit, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, upper_limit, n)
    else:
        raise ValueError("method must be 'trapezoid' or 'simpson'")

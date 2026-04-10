import numpy as np

G = 6.674e-11

def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = (bx - ax) / 2 * nodes + (bx + ax) / 2
    y = (by - ay) / 2 * nodes + (by + ay) / 2
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    w_grid = np.outer(weights, weights)
    f_vals = func(x_grid, y_grid)
    jacobian = (bx - ax) * (by - ay) / 4
    integral = np.sum(f_vals * w_grid) * jacobian
    return integral

def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    sigma = M_plate / (L ** 2)
    def integrand(x, y):
        r_sq = x ** 2 + y ** 2 + z ** 2
        return 1.0 / (r_sq ** 1.5)
    integral = gauss_legendre_2d(integrand, -L/2, L/2, -L/2, L/2, n)
    Fz = G * sigma * m_particle * z * integral
    return Fz

def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    Fz_array = np.array([plate_force_z(z, L, M_plate, m_particle, n) for z in z_values])
    return Fz_array

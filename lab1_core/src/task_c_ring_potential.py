import numpy as np

def ring_potential_point(x: float, y: float, z: float,
                         a: float = 1.0, q: float = 1.0,
                         n_phi: int = 720) -> float:
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    r = np.sqrt((x - a * cos_phi) ** 2 +
                (y - a * sin_phi) ** 2 +
                z ** 2)
    return q * np.mean(1.0 / r)


def ring_potential_grid(y_grid, z_grid,
                        x0: float = 0.0, a: float = 1.0,
                        q: float = 1.0, n_phi: int = 720):
    # 兼容一维坐标向量或二维网格输入
    if y_grid.ndim == 1 and z_grid.ndim == 1:
        Y, Z = np.meshgrid(y_grid, z_grid, indexing='xy')
    else:
        Y, Z = y_grid, z_grid

    shape = Y.shape
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    n_points = y_flat.size

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x_diff = x0 - a * cos_phi
    y_diff = y_flat[:, None] - a * sin_phi
    z_diff = z_flat[:, None]

    r = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
    integrand = 1.0 / r
    V_flat = q * np.mean(integrand, axis=1)
    return V_flat.reshape(shape)


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    return q / np.sqrt(a * a + z * z)

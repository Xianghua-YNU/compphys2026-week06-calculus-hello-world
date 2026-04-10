import numpy as np
import matplotlib.pyplot as plt

def ring_potential_point(x: float, y: float, z: float,
                         a: float = 1.0, q: float = 1.0,
                         n_phi: int = 720) -> float:
    """
    计算均匀带电圆环在单点 (x,y,z) 处的电势
    圆环位于 xy 平面，圆心在原点，半径 a，总电荷等效参数 q
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # 距离 r = sqrt( (x - a cosφ)^2 + (y - a sinφ)^2 + z^2 )
    r = np.sqrt((x - a * cos_phi) ** 2 +
                (y - a * sin_phi) ** 2 +
                z ** 2)
    integrand = 1.0 / r
    
    # 积分 = 均值 * 2π，电势 = q/(2π) * 积分 = q * mean(1/r)
    integral = np.mean(integrand) * 2 * np.pi
    return q * integral / (2 * np.pi)


def ring_potential_grid(y_grid: np.ndarray, z_grid: np.ndarray,
                        x0: float = 0.0, a: float = 1.0,
                        q: float = 1.0, n_phi: int = 720) -> np.ndarray:
    """
    在 yz 平面上（固定 x = x0）计算电势矩阵
    y_grid, z_grid: 形状相同的二维坐标网格
    """
    shape = y_grid.shape
    y_flat = y_grid.ravel()
    z_flat = z_grid.ravel()
    n_points = y_flat.size
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # 广播计算：每个空间点到每个环上点的距离
    x_diff = x0 - a * cos_phi               # (n_phi,)
    y_diff = y_flat[:, None] - a * sin_phi  # (n_points, 1) - (n_phi,) -> (n_points, n_phi)
    z_diff = z_flat[:, None]                # (n_points, 1)
    
    r = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)  # (n_points, n_phi)
    integrand = 1.0 / r
    
    # 对 φ 方向取平均即为电势
    V_flat = q * np.mean(integrand, axis=1)  # (n_points,)
    return V_flat.reshape(shape)


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """轴上 (0,0,z) 电势解析解，用于检验"""
    return q / np.sqrt(a * a + z * z)


if __name__ == "__main__":
    # --- 创建 yz 平面网格 ---
    y_vals = np.linspace(-3, 3, 200)
    z_vals = np.linspace(-3, 3, 200)
    Y, Z = np.meshgrid(y_vals, z_vals)
    
    # --- 计算电势（x=0 平面） ---
    V = ring_potential_grid(Y, Z, x0=0.0, a=1.0, q=1.0)
    
    # --- 计算电场 E = -∇V (使用中心差分) ---
    dy = y_vals[1] - y_vals[0]
    dz = z_vals[1] - z_vals[0]
    Ey, Ez = np.gradient(-V, dy, dz)   # 注意 gradient 返回的次序
    
    # --- 绘图 ---
    plt.figure(figsize=(8, 6))
    
    # 等势线
    levels = np.linspace(0.2, 1.0, 15)
    contour = plt.contour(Y, Z, V, levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    
    # 电场矢量（适当稀疏化）
    skip = 10
    plt.quiver(Y[::skip, ::skip], Z[::skip, ::skip],
               Ey[::skip, ::skip], Ez[::skip, ::skip],
               color='red', alpha=0.7)
    
    # 标记圆环在 yz 平面上的截点 (y = ±a, z = 0)
    plt.scatter([-1, 1], [0, 0], c='blue', marker='o', s=60, 
                label='Ring cross-section (±a,0)')
    
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Equipotential lines and Electric field (yz plane)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- 可选：测试轴上电势与解析解对比 ---
    test_z = 2.0
    print(f"数值计算 V(0,0,{test_z}) = {ring_potential_point(0,0,test_z):.6f}")
    print(f"解析解 V_axis({test_z})   = {axis_potential_analytic(test_z):.6f}")

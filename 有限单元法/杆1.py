"""
轴力杆单元有限元程序
- 杆长 L=3，1 个单元（2 个节点）
- 材料：E=1000, A=1.0
- 左端固定，右端受水平力 F=2
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面时也可保存图片
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定本机中文字体文件路径（Mac 常见路径，取第一个存在的）
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # Mac 自带，支持中文
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Microsoft/SimHei.ttf",  # 若安装过 Office
]
_font_path = None
for _p in FONT_CANDIDATES:
    if os.path.isfile(_p):
        _font_path = _p
        break
if _font_path:
    font_manager.fontManager.addfont(_font_path)
    _prop = font_manager.FontProperties(fname=_font_path)
    plt.rcParams["font.family"] = _prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
else:
    plt.rcParams["axes.unicode_minus"] = False
    # 未找到中文字体时仍用默认，图中中文可能显示为方框

# ========== 1. 问题参数 ==========
L = 3.0          # 杆长度
E = 1000.0       # 弹性模量
A = 1.0          # 截面积
F_right = 2.0    # 右端水平力
n_elem = 1       # 单元数
n_node = n_elem + 1  # 节点数 (2个节点)

# ========== 2. 单元刚度矩阵 ==========
def bar_element_stiffness(E, A, L):
    """
    一维轴力杆单元刚度矩阵（2节点，每节点1个自由度）
    k = (E*A/L) * [[ 1, -1],
                   [-1,  1]]
    """
    k_factor = E * A / L
    Ke = k_factor * np.array([[1.0, -1.0],
                              [-1.0, 1.0]])
    return Ke

Ke = bar_element_stiffness(E, A, L)
print("单元刚度矩阵 Ke (2×2):")
print(Ke)
print()

# ========== 3. 组装总体刚度矩阵 ==========
# 总体自由度：2 个节点 × 1 个位移 = 2
K_global = np.zeros((n_node, n_node))
# 单元 0：节点 0 — 节点 1
K_global[0:2, 0:2] += Ke
print("总体刚度矩阵 K:")
print(K_global)
print()

# ========== 4. 荷载向量 ==========
F_global = np.zeros(n_node)
F_global[1] = F_right   # 右端节点(节点1)受水平力 2
print("总体荷载向量 F:")
print(F_global)
print()

# ========== 5. 边界条件：左端固定 u0 = 0 ==========
# 已知位移：dof 0 固定
fixed_dof = [0]
free_dof = [1]

K_reduced = K_global[np.ix_(free_dof, free_dof)]
F_reduced = F_global[free_dof].copy()

# 求解自由自由度位移：K_reduced * u_free = F_reduced
u_reduced = np.linalg.solve(K_reduced, F_reduced)

# 组装完整位移向量
U = np.zeros(n_node)
U[fixed_dof] = 0.0
U[free_dof] = u_reduced

print("位移解 U (节点位移):")
for i in range(n_node):
    print(f"  节点 {i}: u = {U[i]:.6f}")
print()

# 右端位移理论值: u = F*L/(E*A) = 2*3/1000 = 0.006
print("理论校核: u_right = F*L/(E*A) = {:.6f}".format(F_right * L / (E * A)))
print()

# ========== 6. 绘制变形图 ==========
# 节点坐标（水平杆，从左到右）
x_original = np.array([0.0, L])
x_deformed = x_original + U

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# 放大系数便于观察变形（变形很小）
scale = 50  # 变形放大倍数
x_deformed_scaled = x_original + U * scale

ax1.plot(x_original, np.zeros_like(x_original), 'b-o', linewidth=2, markersize=10, label='变形前')
ax1.plot(x_deformed_scaled, np.zeros_like(x_deformed_scaled), 'r-s', linewidth=2, markersize=10, label=f'变形后 (×{scale})')
ax1.set_xlabel('横坐标 x')
ax1.set_ylabel('纵坐标 (变形已放大)')
ax1.set_title('轴力杆变形图（变形放大 {} 倍）'.format(scale))
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.set_ylim(-0.5, 0.5)

# 第二子图：真实比例
ax2.plot(x_original, np.zeros_like(x_original), 'b-o', linewidth=2, markersize=10, label='变形前')
ax2.plot(x_deformed, np.zeros_like(x_deformed), 'r-s', linewidth=2, markersize=10, label='变形后(真实)')
ax2.set_xlabel('横坐标 x')
ax2.set_ylabel('纵坐标')
ax2.set_title('轴力杆变形图（真实比例）')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.01, 0.01)

plt.tight_layout()
plt.savefig('bar_deformation.png', dpi=150, bbox_inches='tight')
print("变形图已保存为 bar_deformation.png")
if matplotlib.get_backend().lower() != "agg":
    plt.show()


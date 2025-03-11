import matplotlib.pyplot as plt
import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点
G.add_node("x")
G.add_node("A")
G.add_node("a")
G.add_node("B")
G.add_node("b")
G.add_node("C")
G.add_node("y")

# 添加边
G.add_edges_from(
    [("x", "A"), ("A", "a"), ("a", "B"), ("B", "b"), ("b", "C"), ("C", "y")]
)

# 添加导数节点
G.add_node("dy/dx")
G.add_node("dy/da")
G.add_node("dy/db")
G.add_node("dy/dy")

# 添加导数边
G.add_edges_from(
    [
        ("dy/dy", "C'(b)"),
        ("C'(b)", "dy/db"),
        ("dy/db", "B'(a)"),
        ("B'(a)", "dy/da"),
        ("dy/da", "A'(x)"),
        ("A'(x)", "dy/dx"),
    ]
)

# 设置布局
pos = {
    "x": (0, 0.04),
    "A": (1, 0.04),
    "a": (2, 0.04),
    "B": (3, 0.04),
    "b": (4, 0.04),
    "C": (5, 0.04),
    "y": (6, 0.04),
    "dy/dx": (0, 0.01),
    "A'(x)": (1, 0.01),
    "dy/da": (2, 0.01),
    "B'(a)": (3, 0.01),
    "dy/db": (4, 0.01),
    "C'(b)": (5, 0.01),
    "dy/dy": (6, 0.01),
}

# 绘制图形
plt.figure(figsize=(11, 3))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightgrey",
    font_size=10,
    font_weight="bold",
    arrowsize=20,
)
plt.title("Chain Rule Diagram")
plt.savefig("chain_rule_diagram.png")
plt.show()

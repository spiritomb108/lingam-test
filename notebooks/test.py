#%%
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
import lingam
from lingam import DirectLiNGAM
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

# %%
def noise_to_causal_data(
    X: pd.DataFrame,
    G: nx.DiGraph,
) -> pd.DataFrame:
    """
    ノイズX, 因果グラフGからデータセットを生成
    """

    out = X.copy()
    for v in nx.topological_sort(G):
        for u, d in G.pred[v].items():
            out[v] += d["weight"] * out[u]

    return out
# %%
# 因果グラフGtrueの定義
Gtrue = nx.DiGraph()
Gtrue.add_nodes_from([rf"$x_{i}$" for i in range(1, 5)])
Gtrue.add_weighted_edges_from(
    [
        ("$x_1$", "$x_3$", -1),
        ("$x_2$", "$x_1$", 1),
        ("$x_2$", "$x_3$", 2),
        ("$x_2$", "$x_4$", 1),
    ]
)

def get_causal_graph_objects(
    G: nx.DiGraph
) -> Tuple[dict, list]:

    e_label_dict = {(i, j): f'{d["weight"]:.1f}' for i, j, d in G.edges(data=True)}
    e_width_list = [abs(d["weight"]) for (_, _, d) in G.edges(data=True)]

    return e_label_dict, e_width_list

e_label_dict_true, e_width_list_true = get_causal_graph_objects(Gtrue)
#%%
# %%
# データセットの生成
n = 1000
rng = np.random.default_rng(42)

# 各ノードに一様分布ノイズを加える
noise = pd.DataFrame(
    rng.uniform(low=-1.0, high=1.0, size=(n, 4)), columns=Gtrue.nodes()
)

data = noise_to_causal_data(noise, Gtrue)

# %%
l = DirectLiNGAM()
l.fit(data)

# %%
# グラフにする
def get_graph(adjacency_matrix: np.array):
    G = nx.from_numpy_array(
        adjacency_matrix,
        create_using=nx.MultiDiGraph()
    )

    G = nx.relabel_nodes(G, {v: rf"$x_{i+1}$" for i, v in enumerate(G.nodes())})
    e_label_dict_est, e_width_list_est = get_causal_graph_objects(G)

    return G, e_label_dict_est, e_width_list_est

pos = {"$x_1$": (-1, 1), "$x_2$": (1, 1), "$x_3$": (-1, -1), "$x_4$": (1, -1)}

G, e_label_dict_est, e_width_list_est = get_graph(l.adjacency_matrix_.T)

# %%
# viz
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

def plot_causal_graph(ax, G, pos, e_width_list, e_label_dict):

    # ノードの描写
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=1000, node_color="#ffaaaa", edgecolors="k"
    )
    # エッジの描写
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=e_width_list,
        style="solid",
        arrowstyle="->",
        node_size=1000,
        arrowsize=20,
    )
    # ラベリング
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=e_label_dict, ax=ax, rotate=False
    )

ax[0].set(title="DirectLiNGAM")
plot_causal_graph(ax[0], G, pos, e_width_list_est, e_label_dict_est)

ax[1].set(title="True")
plot_causal_graph(ax[1], Gtrue, pos, e_width_list_true, e_label_dict_true)

fig.show()
# %%
# 隣接行列の確認
l.adjacency_matrix_
# %%
# 総合効果 (直接効果に間接効果も考慮したもの) の計算
l.estimate_total_effect(data, from_index=1, to_index=2)
# %%
# lingamライブラリのbootstrapを試す
bs_result = l.bootstrap(data, n_sampling=100)
print(type(bs_result))
# %%
# bootstrapサンプル10個の可視化
pos = {"$x_1$": (-1, 1), "$x_2$": (1, 1), "$x_3$": (-1, -1), "$x_4$": (1, -1)}

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.ravel()

for i, ax in enumerate(axes):
    ax.set_title(f"Resample_{i}")
    G, e_label_dict, e_width_list = get_graph(bs_result.adjacency_matrices_[i].T)
    plot_causal_graph(ax, G, pos, e_width_list, e_label_dict)
#%%
# エッジごとのbootstrap内出現数を計算
pl.DataFrame(bs_result.get_causal_direction_counts())

# %%
# 総合効果の出現確率と、総合効果が出現したときの条件付き中央値(effect)の算出
pl.DataFrame(bs_result.get_total_causal_effects())

# %%
# 総合効果の生じているパス別

def get_all_paths(bs_result:lingam.bootstrap.BootstrapResult) -> pl.DataFrame():
    """
    効果の生じているパス別の、bootstrap中での出現確率と、総合効果の条件付き中央値を出力
    """

    df = pl.DataFrame()
    n_nodes = bs_result.adjacency_matrices_[0].shape[0]
    node_idx = np.arange(n_nodes)
    for i in node_idx:
        for j in node_idx:
            if i==j:
                continue

            data = pl.DataFrame(bs_result.get_paths(from_index=i, to_index=j))

            if data.shape[0] == 0:
                continue

            if df.shape[0] == 0:
                df = data

            df = pl.concat([df, data], how="diagonal_relaxed")

    df = df.unique().sort("probability", descending=True)

    return df

get_all_paths(bs_result)

# %%
# ノイズの独立性検定

def test_noise_dependency(lingam: DirectLiNGAM, data:pl.DataFrame, alpha:float=0.05) -> np.array:
    """
    dataからノード別ノイズの独立性を1:1で検定する
    p-valの計算にはHSICを利用、Bonferroniを適用
    """
    pvals = lingam.get_error_independence_p_values(data)
    n_nodes = pvals.shape[0]
    
    alpha_modified = alpha/((n_nodes**2 - n_nodes)/2)

    rows, cols = np.triu_indices(n_nodes, k=1)
    mask = pvals[rows, cols] < alpha

    return np.column_stack((rows[mask], cols[mask]))

test_noise_dependency(l, data)
# %%
# ノイズどうしが従属しているとき、未知の交絡因子の存在が示唆される
# 交絡因子x_2をあえて落としてノイズの独立性検定
l_missing = lingam.DirectLiNGAM()
l_missing.fit(data[["$x_1$", "$x_3$", "$x_4$"]])
test_noise_dependency(l_missing, data[["$x_1$", "$x_3$", "$x_4$"]])


# %%
# ノイズの非正規性を確認

def plot_error_normality(data:pl.DataFrame) -> plt.figure:
    """
    データ中の各変数の残差の正規性を、DirectLiNGAMの隣接行列により相互作用成分を除去したうえでShapiro-Wilk検定により確認
    """

    l = DirectLiNGAM().fit(data)

    # 各要素の相互作用影響を含めた残差
    data_c = data.values - data.values.mean(axis=0)
    # 残差を(逆)隣接行列に通して他変数由来の成分を抽出・除外
    error = data_c - (l.adjacency_matrix_ @ data_c.T).T

    n_feature = data.shape[1]
    fig, ax = plt.subplots(1, n_feature, figsize=(4*n_feature, 4), sharey=True)
    ax = ax.ravel()

    for i in np.arange(n_feature):
        pval = stats.shapiro(error[:, i])[1].item()
        ax[i].hist(
            error[:, i],
            bins="stone",
            density=True,
            alpha=0.3,
        )

        # 標本統計量を用いた正規分布の密度関数
        xlim = ax[i].get_xlim()
        x_pdf = np.linspace(xlim[0], xlim[1], 1000)
        ax[i].plot(
            x_pdf,
            stats.norm.pdf(x_pdf, loc=error[:, i].mean(), scale=error[:, i].std()),
            c="r"
        )

        ax[i].set(
            title=f"{data.columns[i]} p: {pval:.3f}"
        )

    return fig

plot_error_normality(data).show()

# %%
# グラフに関する事前知識を投入可能
# j->iのノードが.. 1: 存在する / 0: 存在しない / -1: わからない

Aknw_default = (1 - np.diag(np.ones(data.shape[1])))*(-1)
Aknw_default
# %%

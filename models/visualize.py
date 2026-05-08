import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


def visualize_tsne(
    features,
    labels,
    title,
    save_path=None,
    point_size=20,
    max_classes=5,
    proto_features=None,
    proto_labels=None
):
    # ===== 1. 只保留 max_classes 个类别（关键！！）=====
    unique_labels = np.unique(labels)

    if len(unique_labels) > max_classes:
        np.random.seed(42)  # 保证可复现
        selected_classes = np.random.choice(unique_labels, max_classes, replace=False)
    else:
        selected_classes = unique_labels

    # 过滤 features
    mask = np.isin(labels, selected_classes)
    features = features[mask]
    labels = labels[mask]

    # 过滤 prototype（必须同步）
    if proto_features is not None:
        proto_mask = np.isin(proto_labels, selected_classes)
        proto_features = proto_features[proto_mask]
        proto_labels = proto_labels[proto_mask]

    # ===== 2. 拼接 TSNE 输入 =====
    tsne_input = features
    if proto_features is not None:
        tsne_input = np.concatenate([features, proto_features], axis=0)

    # ===== 3. TSNE =====
    tsne_result = TSNE(
        n_components=2,
        random_state=42,
        perplexity=10
    ).fit_transform(tsne_input)

    # ===== 4. 拆分 =====
    if proto_features is not None:
        feat_2d = tsne_result[:len(features)]
        proto_2d = tsne_result[len(features):]
    else:
        feat_2d = tsne_result

    # ===== 5. 颜色 =====
    unique_labels = np.unique(labels)
    cmap = get_cmap('tab10') if len(unique_labels) <= 10 else get_cmap('tab20')

    # ===== 6. 坐标范围 =====
    x_min, x_max = np.min(tsne_result[:, 0]), np.max(tsne_result[:, 0])
    y_min, y_max = np.min(tsne_result[:, 1]), np.max(tsne_result[:, 1])

    x_ticks = np.linspace(x_min, x_max, num=6)
    y_ticks = np.linspace(y_min, y_max, num=6)

    plt.figure(figsize=(6, 5))

    # ===== 7. 普通点（清晰关键）=====
    for j, ulab in enumerate(unique_labels):
        idx = labels == ulab
        color = cmap(j % cmap.N)

        plt.scatter(
            feat_2d[idx, 0],
            feat_2d[idx, 1],
            s=point_size,
            alpha=0.7,               # ↓ 降低透明度更清晰
            color=color,
            linewidths=0.9
        )

    # ===== 8. prototype（突出）=====
    if proto_features is not None:
        for j, ulab in enumerate(proto_labels):
            color = cmap(j % cmap.N)

            plt.scatter(
                proto_2d[j, 0],
                proto_2d[j, 1],
                marker='*',
                s=100,                # ⭐ 更大一点
                facecolors=color,
                edgecolors='red',
                linewidths=1.5,
                zorder=10
            )

    # ===== 9. 标题 =====
    plt.title(title, fontsize=14)

    # ===== 10. 坐标 =====
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tick_params(labelsize=8)

    # ===== 11. legend（稳定显示版本）=====
    legend_elements = []

    for j, ulab in enumerate(unique_labels):
        color = cmap(j % cmap.N)
        legend_elements.append(
            Line2D([0], [0],
                   marker='o',
                   linestyle='None',
                   label=f'Class {ulab}',
                   markerfacecolor=color,
                   markeredgecolor=color,
                   markersize=6)
        )

    if proto_features is not None:
        legend_elements.append(
            Line2D([0], [0],
                   marker='*',
                   linestyle='None',
                   label='Prototype',
                   markerfacecolor='gray',
                   markeredgecolor='red',
                   markersize=10)
        )

    plt.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),   # ⭐ 放到图外（绝对不会消失）
        fontsize=8,
        frameon=True
    )

    # ===== 12. 保存 =====
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.3          # ⭐ 防止legend被裁掉
        )
    else:
        plt.show()

    plt.close()
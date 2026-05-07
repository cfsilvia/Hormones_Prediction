import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
shap_cmap = LinearSegmentedColormap.from_list(
    "shap_red_blue",
    ["#1E88E5", "#FFFFFF", "#FF0052"]  # blue → white → red
)


# --------------------------------------------------
# Core: gradient violin (SHAP-style, no dots)
# --------------------------------------------------
def draw_gradient_violin(ax, data, feat, pos,
                         n_bins=50,
                         max_violin_width=0.4,
                         cmap=None):

    # --- safety (important)
    if len(data) < 2 or np.std(data) == 0:
        ax.plot([np.mean(data), np.mean(data)],
                [pos - 0.2, pos + 0.2],
                color='gray', linewidth=2)
        return

    kde = gaussian_kde(data)

    x_range = np.linspace(data.min(), data.max(), n_bins)
    density = kde(x_range)
    density_max = density.max()

    # 🔥 SHAP-style normalization
    vmin = np.percentile(feat, 5)
    vmax = np.percentile(feat, 95)
    if vmin == vmax:
        vmin, vmax = np.min(feat), np.max(feat)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_bins - 1):
        x_start, x_end = x_range[i], x_range[i + 1]

        dens_val = float(kde((x_start + x_end) / 2))
        half_height = (dens_val / density_max) * max_violin_width

        idx = np.where((data >= x_start) & (data < x_end))[0]
        if len(idx) == 0:
            continue

        # SHAP-like: median feature value → color
        val = np.median(feat[idx])
        color = cmap(norm(val))

        ax.add_patch(
            patches.Rectangle(
                (x_start, pos - half_height),
                x_end - x_start,
                2 * half_height,
                color=color,
                linewidth=0,
                alpha=1.0   # SHAP is quite solid
            )
        )


# --------------------------------------------------
# Main plotting function (single group)
# --------------------------------------------------
def plot_single_group_violin(feature_names,
                             shap_values,
                             feature_values,
                             group_name="",
                             max_features=None,
                             contour_color="black"):

    if max_features is None:
        max_features = len(feature_names)

    feature_names = feature_names[:max_features]
    shap_values = shap_values[:, :max_features]
    feature_values = feature_values.iloc[:, :max_features]

    fig, ax = plt.subplots(figsize=(10, max_features * 1.2 + 2))

    cmap = plt.get_cmap("viridis")

    for i in range(max_features):
        data = shap_values[:, i]
        feat = feature_values.iloc[:, i].to_numpy()

        cmap = shap_cmap

        draw_gradient_violin(
    ax,
    data,
    feat,
    pos=i,
    cmap=cmap
)
    # axis formatting
    ax.set_yticks(range(max_features))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"SHAP Violin Plot ({group_name})")

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
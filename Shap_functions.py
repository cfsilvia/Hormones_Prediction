import shap
import numpy as np
import matplotlib.pyplot as plt

def custom_summary_plot(shap_values, features, feature_names=None, feature_order=None, max_display=None, plot_type="dot"):
    """
    Custom SHAP summary plot where you can control feature order and plotting.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Convert features to numpy if it's a DataFrame
    if hasattr(features, "values"):
        features = features.values

    if feature_names is None:
        feature_names = ["Feature "+str(i) for i in range(features.shape[1])]
    
    num_features = shap_values.shape[1]

    if feature_order is None:
        # By default shap sorts by mean(|shap_value|)
        feature_order = np.arange(num_features)
    else:
        feature_order = np.array([feature_names.index(f) for f in feature_order])

    if max_display is None:
        max_display = num_features

    feature_order = feature_order[:max_display]

    # Now plotting
    plt.figure(figsize=(8, max_display * 0.4))

    for pos, i in enumerate(feature_order):
        print(i)
        shap_violin_plot(
            shap_values[:, i],
            features[:, i],
            feature_names[i],
            pos
        )

    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=12)
    plt.gca().tick_params(labelsize=12)
    plt.xlabel("SHAP value", fontsize=13)
    plt.grid(True, axis='x')
    plt.title("Custom SHAP Summary Plot", fontsize=14)

    # Colorbar
    # import matplotlib as mpl
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label='Feature value (normalized)')
    plt.tight_layout()
    plt.show()


def shap_violin_plot(shap_values, feature_values, feature_name, y_pos):
    """
    Helper function to plot one feature violin (smooth and colored).
    """
    import scipy.stats

    # Remove NaNs
    not_nan = ~np.isnan(shap_values)
    shap_values = shap_values[not_nan]
    feature_values = feature_values[not_nan]

    # Normalize feature values for color
    normed = (feature_values - np.nanmin(feature_values)) / (np.nanmax(feature_values) - np.nanmin(feature_values) + 1e-8)
    
    # Evaluate density
    kde = scipy.stats.gaussian_kde(shap_values)
    x = np.linspace(np.min(shap_values), np.max(shap_values), 500)
    dens = kde(x)
    dens /= dens.max()

    width = 0.4

    for xi, di in zip(x, dens):
        # Find the closest feature value
        idx_closest = np.abs(shap_values - xi).argmin()
        color_val = normed[idx_closest]

        plt.fill_betweenx(
            [y_pos - di * width, y_pos + di * width],
            xi, xi,
            color=plt.cm.coolwarm(color_val),
            linewidth=0
        )
"""This scripts contains plot functions."""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from sklearn.metrics import r2_score

# Local modules
import auxiliary

def legend_patch(label, color="none"):
    """Returns the corresponding label with a Patch object for matplotlib legend purposes.
    
    label: str
        Character chain specifying the label
    color: str
        Corresponding color attribute for label

    Returns: tuple -> tuple(str, matplotlib.patches.Patch)
        Specified label with Patch

    """
    return label, mpatches.Patch(color=color, label=label)


def plot(indices, observed, predicted, scale = "linear", mode="plot",
         title="", metric="", xlabel="", ylabel="", alphas=(1, 1),
         xleft=None, xright=None, ytop=None, ybottom=None,
         r2=None, loss=None, normalize=False,
         save_to=None, filename="plot.png", overwrite=True,
         lab_1="observed", lab_2="predicted",
         **kwargs
):
    """Generate a specified figure.
    
    indices: Iterator (numpy.ndarray)
        Observed and predicted indices to perform plots on
        values need to be in array observed/predicted indices range.

    observed: Iterator (numpy.ndarray)
        Observed values

    predicted: Iterator (numpy.ndarray)
        Predicted values

    scale: str
        Chosen scale for observed and predicted values
        "linear", "log", ...

    mode: str
        Selected mode for plot taking values such as 
        "plot" -> line plot
        "bar" -> barplot plot
        "hist" -> histogram plot
        "hist2d" -> histogram2d plot
        "scatter" -> scatter plot
        "violin" -> violin plot

    title, xlabel, ylabel: str
        Title, x-axis and y-axis labels to assign to
        the figure

    xleft, xright: float, optional
        lower and upper x limit

    ybottom, ytop: int, optional
        lower and upper y limit

    alphas: tuple(float), optional
        alpha values for values to plot

    normalize: bool
        Normalize values so that it is bound to [0; 1] values
        or [-1; 1] values if delta=True

    Returns: tuple (matplotlib.figure, matplotlib.axes.Axes)
        Figure and Axes for graphical purposes

    """
    # Plot mode selection
    plotting_mode = {
        "plot": lambda ax: ax.plot,
        **dict.fromkeys(['bar', 'delta_bar'], lambda ax: ax.bar),
        "hist": lambda ax: ax.hist,
        "scatter": lambda ax: ax.scatter,
        "hist2d": lambda ax: ax.hist2d,
        "violin": lambda ax: ax.violinplot
    }
    # Check mode
    if mode not in plotting_mode.keys():
        raise Exception("Selected mode is invalid")

    alphas = (alphas, ) if isinstance(alphas, (int, float)) else tuple(alphas)
    alphas = alphas + (1,)
    delta = True if mode.startswith("delta") else False

    # Optional arguments
    if mode.startswith("hist"):
        kwargs["bins"] = kwargs.get("bins", 100)  # necessary if mode == "hist"/"hist2d"
        kwargs["cmap"] = kwargs.get("cmLoadap", plt.cm.jet)
        kwargs["norm"] = kwargs.get("norm", mcolors.LogNorm())

    xticks = kwargs.get("xticks", None)
    yticks = kwargs.get("yticks", None)
    grid = kwargs.get("grid", False)

    # Conversion
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    if not isinstance(observed, np.ndarray):
        observed = np.array(observed)

    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)

    # Metrics
    r2 = r2 if r2 else r2_score(observed, predicted)
    loss = loss if loss else ((observed - predicted) ** 2).mean()

    if normalize:
        # Normalization needed before mean, std, median calculation
        op_concat = np.concatenate((observed, predicted), axis=0)
        if not delta:
            op_concat = auxiliary.normalization_min_max(all, 0, 1)
            observed = op_concat[:observed.shape[0]]
            predicted = op_concat[observed.shape[0]:]

    # Observed and Predicted : Mean, std, median
    mean_observed, std_observed = observed.mean(), observed.std()
    mean_predicted, std_predicted = predicted.mean(), predicted.std()
    median_observed, median_predicted = np.median(observed), np.median(predicted)

    # Delta (Observed - Predicted)
    delta_values = observed - predicted
    delta_values = auxiliary.normalization_min_max(delta_values, -1, 1) if normalize else delta_values

    # Delta (Observed - Predicted) : Mean, std, median
    mean_delta, std_delta = delta_values.mean(), delta_values.std()
    median_delta = np.median(delta_values)

    # Plot depending on mode
    fig, ax = plt.subplots(figsize=(8, 7))
    label_observed, label_predicted = lab_1, lab_2
    if (mode == "plot"):
        plotting_mode[mode](ax)(indices, observed[indices], label=label_observed, alpha=alphas[0], **kwargs)
        plotting_mode[mode](ax)(indices, predicted[indices], label=label_predicted, alpha=alphas[1], **kwargs)
    elif (mode == "hist"):
        bins=kwargs["bins"]
        plotting_mode[mode](ax)(observed[indices], bins=bins, label=label_observed, alpha=alphas[0])
        plotting_mode[mode](ax)(predicted[indices], bins=bins, label=label_predicted, alpha=alphas[1])
    elif (mode == "bar"):
        plotting_mode[mode](ax)(indices, observed[indices], label=label_observed, alpha=alphas[0], **kwargs)
        plotting_mode[mode](ax)(indices, predicted[indices], label=label_predicted, alpha=alphas[1], **kwargs)
    elif (mode == "violin"):
        v1 = plotting_mode[mode](ax)(observed[indices], positions=[0], **kwargs)
        v2 = plotting_mode[mode](ax)(predicted[indices], positions=[0.5], **kwargs)
        labels = []
        labels.append((mpatches.Patch(color=v1["bodies"][0].get_facecolor().flatten()), label_observed))
        labels.append((mpatches.Patch(color=v2["bodies"][0].get_facecolor().flatten()), label_predicted))
        main_legend = ax.legend(*zip(*labels), loc=2)
        ax.add_artist(main_legend)
    elif (mode == "scatter"):
        plotting_mode[mode](ax)(observed[indices], predicted[indices], alpha=alphas[0], **kwargs)
    elif (mode == "hist2d"):
        plotting_mode[mode](ax)(observed[indices], predicted[indices], **kwargs)
    elif (mode == "delta_bar"):
        delta_values_idx = delta_values[indices]
        plotting_mode[mode](ax)(indices, delta_values_idx, alpha=alphas[0], **kwargs)

    # Main Legend
    handles, labels = ax.get_legend_handles_labels()
    if (handles != []) & (labels != []):
        main_legend = ax.legend(handles, labels, loc="upper left")
        ax.add_artist(main_legend)

    # R2 & Loss Legend
    handles_r2loss, labels_r2loss = [], []
    ## R2
    r2_label, r2_patch = legend_patch(f"R2 = {r2:.4f}")
    handles_r2loss.append(r2_patch)
    labels_r2loss.append(r2_label)
    ## Loss
    loss_label, loss_patch = legend_patch(f"loss = {loss:.4f}")
    handles_r2loss.append(loss_patch)
    labels_r2loss.append(loss_label)

    r2_loss_legend = ax.legend(handles_r2loss, labels_r2loss, loc="upper right",
                               handlelength=0, handletextpad=0)

    # Add legend
    ax.add_artist(r2_loss_legend)

    # Mean, std, median Legend
    if delta:
        # Delta : (Observed - Predicted)
        handles_delta, labels_delta = [], []
        mean_delta_label, mean_delta_patch = legend_patch(f"mean = {mean_delta:.3f}")
        std_delta_label, std_delta_patch = legend_patch(f"std = {std_delta:.3f}")
        median_delta_label, median_delta_patch = legend_patch(f"median = {median_delta:.3f}")
    
        handles_delta.extend([mean_delta_patch, std_delta_patch, median_delta_patch])
        labels_delta.extend([mean_delta_label, std_delta_label, median_delta_label])
    
        msm_delta_legend = fig.legend(handles_delta, labels_delta, title=f"Delta{metric}",
                                        handlelength=0, handletextpad=0, borderaxespad=0,
                                        bbox_to_anchor=(1.11, 0.88))
        # Add legend
        ax.add_artist(msm_delta_legend)
    else:
        # Observed
        handles_obs, labels_obs = [], []
        mean_observed_label, mean_observed_patch = legend_patch(f"mean = {mean_observed:.3f}")
        std_observed_label, std_observed_patch = legend_patch(f"std = {std_observed:.3f}")
        median_observed_label, median_observed_patch = legend_patch(f"median = {median_observed:.3f}")
    
        handles_obs.extend([mean_observed_patch, std_observed_patch, median_observed_patch])
        labels_obs.extend([mean_observed_label, std_observed_label, median_observed_label])
    
        msm_observed_legend = fig.legend(handles_obs, labels_obs, title="observed",
                                            handlelength=0, handletextpad=0, borderaxespad=0,
                                            bbox_to_anchor=(1.06, 0.88))

        # Add legend
        ax.add_artist(msm_observed_legend)

        # Predicted
        handles_pred, labels_pred = [], []
        mean_predicted_label, mean_predicted_patch = legend_patch(f"mean = {mean_predicted:.3f}")
        std_predicted_label, std_predicted_patch = legend_patch(f"std = {std_predicted:.3f}")
        median_predicted_label, median_predicted_patch = legend_patch(f"median = {median_predicted:.3f}")

        handles_pred.extend([mean_predicted_patch, std_predicted_patch, median_predicted_patch])
        labels_pred.extend([mean_predicted_label, std_predicted_label, median_predicted_label])
    
        msm_predicted_legend = fig.legend(handles_pred, labels_pred, title="predicted",
                                            handlelength=0, handletextpad=0, borderaxespad=0,
                                            bbox_to_anchor=(1.06, 0.7))

        # Add legend
        ax.add_artist(msm_predicted_legend)

    # Scale selected by user
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    # Option for specific mode
    if (mode == "scatter"):
        title += f" - scale={scale}"
        # Range to have xlim=ylim
        xy_lim = auxiliary.min_max(ax.get_xlim() + ax.get_ylim())
        ax.set_xlim(xy_lim)
        ax.set_ylim(xy_lim)
    elif (mode == "delta_bar"):
        # Horizontal separation line
        ax.axhline(y=0, color='orange', linestyle = '--')
        # Range to have ylim(a, b) with a=max(absolute(y)) and a=b
        highest_value = max(np.abs(ax.get_ylim()))
        ax.set_ylim((-highest_value, highest_value))

    # Set figure label, limit and legend
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Label
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Limit in x and y
    ax.set_xlim(left=xleft, right=xright)
    ax.set_ylim(top=ytop, bottom=ybottom)
    # Grid
    ax.grid(grid)

    if save_to is not None:
        if not auxiliary.isdir(save_to):
            raise Exception("Specified directory does not exists")

        root, _ = os.path.splitext(filename)
        root = root if root.isalnum() else "plot"
        # Save file to
        save_to = auxiliary.to_dirpath(save_to)
        filename = auxiliary.replace_extension(root, "png")
        filepath = save_to + filename if overwrite else \
                   auxiliary.filepath_with_suffix(save_to + filename)

        plt.savefig(filepath, bbox_inches = 'tight')

    return fig, ax


if __name__ == "__main__":
    indices = [0, 1, 2]
    obs = [1, 2, 2]
    pred = [1.1, 2.2, 3.1]
    plot(indices, obs, pred, mode="plot", save_to="./")

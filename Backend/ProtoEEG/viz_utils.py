from protopnet.spikenet_helpers import (
    EEG_ConcatDataset,
    zscale,
    t,
    channel_names,
    offset,
    eeg_preprocess_for_plotting,
    label_finder,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import torch
from matplotlib.patches import Rectangle

# define a color mpa used for creating heatmap overlays
cmap = LinearSegmentedColormap.from_list("custom", ["lightblue", "darkblue"], N=256)
pink_darkred_cmap = LinearSegmentedColormap.from_list(
    "custom", ["pink", "darkred"], N=256
)
white_green_cmap = LinearSegmentedColormap.from_list(
    "custom", ["white", "darkgreen"], N=256
)



def min_max_normalize(arr):
    min_val = min(arr)
    max_val = max(arr)

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return [0] * len(arr)

    return [(x - min_val) / (max_val - min_val) for x in arr]



def plot_topoplot(fig, ax1, weight, normalized_weights):
    """
    plots the topoplot

    """

    # First subplot: Original plot
    ax1.set_xlim(-1.4, 1.8)
    ax1.set_ylim(-1.2, 1.2)

    # position dict (POSITION OF CHANNEL WE ARE PLOTTING IN OG EEG)
    pos_dict = {
        "Fp1": 0,
        "F3": 1,
        "C3": 2,
        "P3": 3,
        "F7": 4,
        "T3": 5,
        "T5": 6,
        "O1": 7,
        "Fz": 8,
        "Cz": 9,
        "Pz": 10,
        "Fp2": 11,
        "F4": 12,
        "C4": 13,
        "P4": 14,
        "F8": 15,
        "T4": 16,
        "T6": 17,
        "O2": 18,
        "Fp1-F7": 19,
        "F7-T3": 20,
        "T3-T5": 21,
        "T5-O1": 22,
        "Fp2-F8": 23,
        "F8-T4": 24,
        "T4-T6": 25,
        "T6-O2": 26,
        "Fp1-F3": 27,
        "F3-C3": 28,
        "C3-P3": 29,
        "P3-O1": 30,
        "Fp2-F4": 31,
        "F4-C4": 32,
        "C4-P4": 33,
        "P4-O2": 34,
        "Fz-Cz": 35,
        "Cz-Pz": 36,
    }

    # Define the annotations
    annotations = [
        (0.384738, 0.849112, "Fp1"),
        (0.613525, 0.849124, "Fp2"),
        (0.34544, 0.688515, "F3"),
        (0.652782, 0.688515, "F4"),
        (0.314023, 0.4971, "C3"),
        (0.684199, 0.4971, "C4"),
        (0.345406, 0.305717, "P3"),
        (0.652817, 0.305717, "P4"),
        (0.384731, 0.145013, "O1"),
        (0.613491, 0.145013, "O2"),
        (0.199637, 0.714664, "F7"),
        (0.798585, 0.714664, "F8"),
        (0.128936, 0.497074, "T3"),
        (0.869287, 0.497074, "T4"),
        (0.199618, 0.279469, "T5"),
        (0.798604, 0.279469, "T6"),
        (0.499111, 0.682161, "Fz"),
        (0.499111, 0.497121, "Cz"),
        (0.499111, 0.312019, "Pz"),
    ]
    # Convert annotations to a dictionary for easy access
    ann_dict = {label: ((x - 0.5) * 2, (y - 0.5) * 2) for x, y, label in annotations}
    # Draw the large circle as the border
    border_circle = plt.Circle((0, 0), 1.0, fill=False, edgecolor="black", linewidth=2)
    ax1.add_artist(border_circle)

    # Define the connections
    connections = [
        ("Fp1", "F7"),
        ("F7", "T3"),
        ("T3", "T5"),
        ("T5", "O1"),
        ("Fp2", "F8"),
        ("F8", "T4"),
        ("T4", "T6"),
        ("T6", "O2"),
        ("Fp1", "F3"),
        ("F3", "C3"),
        ("C3", "P3"),
        ("P3", "O1"),
        ("Fp2", "F4"),
        ("F4", "C4"),
        ("C4", "P4"),
        ("P4", "O2"),
        ("Fz", "Cz"),
        ("Cz", "Pz"),
    ]
    # Draw the lines
    for start, end in connections:
        start_x, start_y = ann_dict[start]
        end_x, end_y = ann_dict[end]

        label = f"{start}-{end}"
        # get weight for current channel
        curr_pos = pos_dict[
            label
        ]  # use the label that's getting plotting and get which channel it is in the weight array

        val = weight[curr_pos]

        val = normalized_weights[curr_pos].item()
        color = cmap(val)

        ax1.plot(
            [start_x, end_x],
            [start_y, end_y],
            color=color,
            linewidth=8,
            alpha=0.9,
            zorder=1,
        )

    # Plot each annotation and draw a filled red circle around it
    for x, y, label in annotations:
        x_new, y_new = ann_dict[label]

        # get weight for current channel
        curr_pos = pos_dict[
            label
        ]  # use the label that's getting plotting and get which channel it is in the weight array

        val = weight[curr_pos]

        val = normalized_weights[curr_pos].item()
        color = cmap(val)

        circle = plt.Circle(
            (x_new, y_new),
            0.13,
            fill=True,
            facecolor=color,
            edgecolor="none",
            alpha=1.0,
            zorder=1,
        )  # Increased circle size
        ax1.add_artist(circle)

        if val > 0.5:
            text_color = "white"
            bolded = "bold"
        else:
            text_color = "black"
            bolded = "normal"

        ax1.annotate(
            label,
            (x_new, y_new),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color=text_color,
            fontweight=bolded,
            fontsize=14,
        )  #

    # Add ears (ovals)

    left_ear = Ellipse((-1.1, 0), 0.2, 0.4, fill=False, edgecolor="black", linewidth=2)
    right_ear = Ellipse((1.1, 0), 0.2, 0.4, fill=False, edgecolor="black", linewidth=2)
    ax1.add_artist(left_ear)
    ax1.add_artist(right_ear)
    # Add nose (two-line nose pointing outward)
    ax1.plot([0, -0.05], [1.02 + 0.05, 0.94 + 0.05], color="black", linewidth=2)
    ax1.plot([0, 0.05], [1.02 + 0.05, 0.94 + 0.05], color="black", linewidth=2)
    # Remove axis ticks and labels for the first subplot
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # Remove axis lines for the first subplot
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    # Set aspect ratio to equal for a perfect circle
    ax1.set_aspect("equal")

    # Create new axis for colorbar below the topoplot
    bbox = ax1.get_position()

    # Calculate the width of the colorbar based on the topoplot's xlim
    total_width = 1.8 - (-1.4)  # = 3.2
    colorbar_width = bbox.width * (3.0 / 3.2)  # Slightly smaller than total width

    # Create new axis for colorbar below the topoplot
    bbox = ax1.get_position()

    # Center the colorbar based on ax1's position
    cbar_ax = fig.add_axes(
        [
            bbox.x0 + bbox.width * 0.1,  # Start at 10% from left edge of ax1
            bbox.y0 - 0.05,  # Below the plot
            bbox.width * 0.68,  # Width is 80% of ax1 width
            0.02,  # Height
        ]
    )

    # Create a normalization object for the colorbar
    norm = plt.Normalize(0, 1)  # normalize to [0,1] since we use normalized weights

    # Create colorbar using the same colormap
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
    )

    # Set colorbar ticks
    cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['0', '0.5', '1'])
    cbar.set_ticklabels(["Less Important \n Channel", "More Important \n Channel"])



def local_analysis(
    model,
    eeg_filename,
    proto_filenames,
    input_eeg_label,
    train_dict,
    test_dict,
    save_filename,
):

    # get the correct EEG
    try:
        eeg_raw = train_dict[eeg_filename]
    except:
        eeg_raw = test_dict[eeg_filename]

    plot_eeg = eeg_preprocess_for_plotting(eeg_raw)
    input_eeg_label = input_eeg_label

    # define the plot
    fig, axs = plt.subplots(
        1,
        len(proto_filenames) + 2,
        figsize=(45, 11),
        gridspec_kw={
            "width_ratios": [2.75, 1.45] + [0.75 for i in range(len(proto_filenames))],
            "wspace": 0.001,
        },
    )

    # iterate over all protos

    input_weight = model.prototype_layer.spikenet_weight_dict[eeg_filename].cpu() / (
        model.prototype_layer.spikenet_weight_dict[eeg_filename].cpu().sum() + 0.000001
    )

    # Add heatmap on top of input eeg
    normalized_weights = min_max_normalize(input_weight)

    plot_topoplot(fig, axs[0], input_weight, normalized_weights)
    axs[0].set_title(f"Topoplot", x=0.44, ha="center", fontsize=24, fontweight="bold")

    # Calculate bar parameters
    bar_width = 0.4  # Maximum width of the bars
    label_position = -0.7  # Position of channel labels
    bar_right_edge = label_position - 0.1  # Space between bars and labels

    for i in range(plot_eeg.shape[0]):

        # if "T6" in channel_names[i] or "C4" in channel_names[i] or "T4" in channel_names[i]:
        #        continue

        axs[1].plot(t, plot_eeg[i, :] * zscale - offset[i], "k", linewidth=1.0)

        # Plot vertical weight bar to the left of labels
        bar_height = normalized_weights[i] * 1.5
        bar_color = cmap(bar_height.item())
        bar_width_scaled = bar_width * bar_height  # Scale width by normalized weight

        axs[1].barh(
            y=-offset[i],  # Center of the bar
            width=bar_width_scaled,  # Width based on weight
            height=0.8,  # Height of the bar
            left=bar_right_edge - bar_width_scaled,  # Position bar so right edge aligns
            color=bar_color,
            alpha=0.7,
            zorder=2,
        )

        axs[1].text(
            label_position,
            -offset[i],
            channel_names[i],
            fontsize=10,
            verticalalignment="center",
        )

    axs[1].text(
        bar_right_edge - 0.65,
        -22,
        "Channel Importance",
        rotation=90,
        ha="center",
        va="center",
        fontsize=18,
    )
    axs[1].set_xlim(bar_right_edge - 0.50, 1.50)
    axs[1].axis("off")
    axs[1].set_ylim(-46.23281208404306, 1.3109161211674076)

    # title is set to the end to include prediction

    # Add heatmap on top of input eeg
    normalized_weights = min_max_normalize(input_weight)

    ###### ADD ALL OTHER Prototype MATCHES #####
    count = 2
    prediction = []
    for proto_index in proto_filenames:

        # process prototype eeg
        proto_filename = proto_filenames[count - 2]
        eeg = train_dict[proto_filename]
        eeg_containing_proto = eeg_preprocess_for_plotting(eeg)
        eeg_containing_proto.max()

        # plot the large EEG
        for i in range(eeg_containing_proto.shape[0]):
            axs[count].plot(
                t, eeg_containing_proto[i, :] * zscale - offset[i], "k", linewidth=1.0
            )

        proto_label = label_finder(proto_filename)

        axs[count].set_xlim(-0.25, 1.25)
        axs[count].axis("off")
        # axs[count].set_title(f"#{count-1} Best Match \n Ground Truth Votes (%): {100*float(proto_label):.2f}")
        axs[count].set_title(
            f"#{count-1} Best Match\nGround Truth\nVotes (%): {100*float(proto_label):.2f}",
            fontweight="bold",
        )
        count += 1
        prediction.append(float(proto_label))

    # Title set here for prediction
    axs[1].set_title(
        f"Model Prediction: {100*sum(prediction)/len(prediction):.2f}%",
        fontweight="bold",
        fontsize=15,
        color="#FF5F1F",
    )

    # similarity score is the max activation of the current
    plt.tight_layout()
    plt.savefig(f"{save_filename}.png", bbox_inches="tight", pad_inches=0.05, dpi=100)
    print(f"Finished and Saved Local Analysis To {save_filename}")
    plt.close("all")

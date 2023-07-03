import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


labels = ["Top-1", "Top-5", "Top-10"]

R4R_insert = [4.95, 6.76, 11.71]
R4R_delete = [96.00, 99.06, 99.76]
R4R_update = [6.72, 15.42, 20.84]
tufano_insert = [22.22, 33.33, 33.33]
tufano_delete = [22.36, 42.93, 52.24]
tufano_update = [7.01, 15.41, 19.26]

codeT5_R4R_insert = [13.51, 16.67, 17.57]
codeT5_R4R_delete = [67.07, 70.35, 71.53]
codeT5_R4R_update = [24.52, 33.75, 36.31]
codeT5_tufano_insert = [38.89, 66.67, 66.67]
codeT5_tufano_delete = [43.65, 64.40, 71.02]
codeT5_tufano_update = [28.46, 42.99, 47.64]

PLBART_R4R_insert = [9.91, 14.41, 16.67]
PLBART_R4R_delete = [67.06, 96.71, 98.12]
PLBART_R4R_update = [19.07, 28.55, 33.36]
PLBART_tufano_insert = [27.78, 38.89, 55.56]
PLBART_tufano_delete = [44.53, 63.51, 71.91]
PLBART_tufano_update = [27.50, 39.68, 42.47]


def plot_graph(plot_type, dataset, base_data, codeT5, PLBART):
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    dataset_label = "R4R_CC" if "Review4Repair" in dataset else dataset
    rects1 = ax.bar(x - width, base_data, width, label=dataset_label)
    rects2 = ax.bar(x, PLBART, width, label="Fine-tuned PLBART")
    rects3 = ax.bar(x + width, codeT5, width, label="Fine-tuned CodeT5")

    y_lim_padding = 40 if plot_type == "Delete" and dataset == "Review4Repair model_cc" else 7
    legend_location = "upper left"

    # Add some text for labels, title and custom x-axis tick labels, etc.
    dataset = "Review4Repair" if "Review4Repair" in dataset else "Tufano"
    # title = f'Performance on {plot_type} samples ({dataset} dataset)' if "Review4Repair" in dataset else f'Performance on {plot_type} samples (Dataset by Tufano et al.)'
    ax.set_ylabel("Accuracy(%)")
    
    # ax.set_ylim(0, max(max(max(codeT5), max(PLBART)), max(base_data)) + y_lim_padding)
    ax.set_ylim(0, 130)

    # Format y-axis labels
    threshold = 100  # Set the threshold value for labels to be shown
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '' if x > threshold else '{:g}'.format(x)))
    
    ax.set_xlabel("Number of Top Predictions")
    
    # ax.set_title(f'Performance on {plot_type} samples ({dataset} dataset)')
    ax.set_xticks(x, labels)
    ax.legend(loc=legend_location)

    ax.bar_label(rects1, padding=5)
    ax.bar_label(rects2, padding=5)
    ax.bar_label(rects3, padding=5)

    # locator = ticker.MaxNLocator(nbins=8)  # Set the number of desired tick intervals
    # ax.yaxis.set_major_locator(locator)
    # ticks = ax.yaxis.get_major_ticks()
    # for idx, tick in enumerate(ticks):
    #     print(tick.label1)
    #     if tick.label1.get_text() == '':
    #         ticks[idx].set_visible(False)
    

    fig.tight_layout()
    # plt.show()
    
    # save the figure
    fig_dir = "figures"
    fig_title = f"R4R_{plot_type.lower()}.png" if "Review4Repair" in dataset else f"tufano_{plot_type.lower()}.png"
    fig_path = f"{fig_dir}/{fig_title}"
    plt.savefig(fig_path)


if __name__ == "__main__":
    samples = [
        ("Insert", "Review4Repair model_cc", R4R_insert, codeT5_R4R_insert, PLBART_R4R_insert),
        ("Delete", "Review4Repair model_cc", R4R_delete, codeT5_R4R_delete, PLBART_R4R_delete),
        ("Update", "Review4Repair model_cc", R4R_update, codeT5_R4R_update, PLBART_R4R_update),
        ("Insert", "Tufano 2-encoder", tufano_insert, codeT5_tufano_insert, PLBART_tufano_insert),
        ("Delete", "Tufano 2-encoder", tufano_delete, codeT5_tufano_delete, PLBART_tufano_delete),
        ("Update", "Tufano 2-encoder", tufano_update, codeT5_tufano_update, PLBART_tufano_update),
    ]

    for sample in samples[:]:
        plot_graph(*sample)

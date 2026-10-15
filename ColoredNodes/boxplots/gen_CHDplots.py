import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    df = pd.read_csv("out_all_algs.csv")
    #df = df[df["Algorithm"] != "GES"]
    df = df.loc[df["Algorithm"].isin(["GreedyColor", "GreedyColor+turn", "GreedyDP", "GreedyDP+turn"])]

    size_options = [5,8,20]
    sample_options = [100, 500, 1000]


    cols = [f"n = {x}" for x in sample_options]
    rows = [f"p = {x}" for x in size_options]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    plt.setp(axes.flat, xlabel='edge probability', ylabel='CHD')

    pad = 5 # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)


    for row in [0,1,2]:
        tmp1_df = df[df["num_nodes"] == size_options[row]]
        tmp1_df = tmp1_df[tmp1_df["num_colors"] == int(size_options[row]/2)]    # Change between 2 and int(size_options[row]/2)

        for col in [0,1,2]:
            plt.subplot(3,3, 3*row+col+1)
            tmp2_df = tmp1_df[tmp1_df["num_samples"] == sample_options[col]]
            sns.boxplot(data=tmp2_df, x="edge_prob", y="CHD", hue="Algorithm", palette="viridis")
  
    plt.show()

if __name__ == "__main__":
    main()
import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from pandas import DataFrame, concat
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


def chart(df: DataFrame, x, y, target) -> alt.Chart:
    """ Turns Pandas DataFrame into Altair Chart Class """
    alt.themes.enable('dark')  # Enable dark mode theme
    graph = alt.Chart(
        df,
        title=f"{y} by {x} for {target}",
    ).mark_circle(size=100).encode(
        x=x,
        y=y,
        color=target,
        tooltip=alt.Tooltip(df.columns.to_list())
    )
    return graph


def damage_calc(damage: str):
    """ Assigns an average numerical value to a Monster's Damage attribute"""

    # Determine Damage Attributes
    if '+' in damage:
        mod, no_mod = damage.split('+')[1], damage.split('+')[0]
        d_type, d_qty = no_mod.split('d')[1], no_mod.split('d')[0]
    else:
        d_type, d_qty, mod = damage.split('d')[1], damage.split('d')[0], 0

    # SUM (Average Dice Roll, Modifier)
    damage_value = int(d_qty) * (int(d_type) / 2) + int(mod)

    return damage_value


def corr_heatmap(df: DataFrame, ordinal: any = None):
    """ Receive Pandas DataFrame and return URL for SNS heatmap image src display"""

    # Initialize the OrdinalEncoder
    oe = OrdinalEncoder()

    # Fit and transform the categorical columns
    df['Type'] = oe.fit_transform(df[['Type']])
    df['Rarity'] = oe.fit_transform(df[['Rarity']])
    if ordinal:
        df['Damage'] = oe.fit_transform(df[['Damage']])  # See Below

    # Apply Damage Calculate Function to Damage Column
    if not ordinal:
        df['Damage'] = df['Damage'].apply(damage_calc)

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Create a triangular matrix
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True

    # Create the heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.style.use('dark_background')  # Dark Mode Style
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True)
    if not ordinal:
        plt.title('Bandersnatch Correlation Heatmap (Damage Custom)')
    else:
        plt.title('Bandersnatch Correlation Heatmap (Damage Ordinal)')

    # Save the image as a PNG in memory
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format="png", bbox_inches="tight")
    buffer2.seek(0)
    plt.close()

    # Encode the image in base64 and return as a string that can be used as a src attribute in an HTML img tag
    image_base64 = base64.b64encode(buffer2.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def bar_chart(x, y, df: DataFrame):
    """ Shows a Damage Bar Chart """

    # Create Set Values to avoid duplicates
    set_x = sorted(set(x.values))

    def find_d(set_list: list, ct: int = None):
        """ Ensure the sorted list starts with the lowest dice quantity """
        while not ct:
            for i in range(len(set_list)):
                d_loc = set_list[i].find('d')
                if d_loc == 1:
                    ct = i
                    if ct:
                        final = set_list[ct:] + set_list[:ct]
                        return final

    # create list of values to search for
    ss_x = find_d(set_x)

    # initialize list to store results
    results = []

    # loop over the list of values and find the first occurrence in the DataFrame
    for dmg in ss_x:
        index = df.loc[df[x.name] == dmg].first_valid_index()
        if index is not None:
            results.append(df.loc[index])

    # create new DataFrame with the results
    results_df = concat(results, axis=1).transpose()
    ss_y = results_df[y.name]
    xy_len = len(results_df) if len(results_df) <= 100 else 100

    # Create the bar chart using Seaborn
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.barplot(x=ss_y.values[:xy_len], y=ss_x[:xy_len])
    plt.style.use('dark_background')  # Dark Mode

    # Set the plot title and axis labels
    plt.title("Average Damage vs. Dice Roll + Modifier")
    plt.xlabel(y.name)
    plt.ylabel(x.name)

    # Save the image as a PNG in memory
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format="png", bbox_inches="tight")
    buffer3.seek(0)
    plt.close()

    # Encode the image in base64 and return as a string that can be used as a src attribute in an HTML img tag
    image_base64 = base64.b64encode(buffer3.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def results_charts(df1, df2=None, df3=None):
    """ Return the Predicted Results and Confidence Values! """

    df1 = df1 if df1 else pd.DataFrame()
    df2 = df2 if df2 else pd.DataFrame()
    df3 = df3 if df3 else pd.DataFrame()

    # create subplots depending on the number of data frames
    fig, axes = plt.subplots(1, len([df for df in [df1, df2, df3] if df]), figsize=(10, 5))

    # convert data frames to list for iteration
    dfs = [df for df in [df1, df2, df3] if not df.empty]

    # plot bar charts for each data frame
    for i, df in enumerate(dfs):
        df.plot.bar(x='Category', y='Value', ax=axes[i], legend=False)
        axes[i].set_title(f'Data Frame {i + 1}')
        axes[i].set_xlabel('Category')
        axes[i].set_ylabel('Value')

    # add space between subplots
    plt.subplots_adjust(wspace=0.4)

    # return combined bar plot
    return plt.show()

    # Set the plot title and axis labels
    # plt.title("Average Damage vs. Dice Roll + Modifier")
    # plt.xlabel(y.name)
    # plt.ylabel(x.name)

    # Save the image as a PNG in memory
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format="png", bbox_inches="tight")
    buffer3.seek(0)
    plt.close()

    # Encode the image in base64 and return as a string that can be used as a src attribute in an HTML img tag
    image_base64 = base64.b64encode(buffer3.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

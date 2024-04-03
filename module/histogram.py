from matplotlib import pyplot as plt
import numpy as np 
import seaborn as sns
from module.getters import getHeadTwitchDf, getExperimentalInfo
from module.stats import processQuantitativeStats, updateQuantitativeStats
from module.utils import figure_cache
from statannotations.Annotator import Annotator



@figure_cache("head_twitch_histogram")
def headTwitchHistogram(
    HT_filename,
    experiment=None,
    vairable=None,
    outlier_test=None,
    p_value_threshold=None,
    from_scratch=None,
):
    HT_data = getHeadTwitchDf(HT_filename)
    data, order, palette = buildHeadTwitchHistogramData(
        HT_filename, experiment, vairable
    )

    title = vairable.replace("_", " at ") + " min"
    ylabel = "events / min"

    # REMI add stats calcultaion to input to histogram builder and outlier same as for quantativeHistograms()
    # the last quantitative test is coded to return the labels directly, thus the need for the bool
    (is_significant, significance_infos, test_results) = processQuantitativeStats(
        getExperimentalInfo(HT_filename)[experiment], data, p_value_threshold
    )

    updateQuantitativeStats(
        HT_filename,
        [
            {
                "data_type": "HT",
                "experiment": experiment,
                "compound": None,
                "region": None,
                **test_result,
            }
            for test_result in test_results
        ],
    )

    fig = buildHistogram(
        title,
        ylabel,
        data,
        order,
        hue='treatment',
        palette=palette,
        significance_infos=significance_infos if is_significant else None,
    )
    return fig


def buildHeadTwitchHistogramData(
    HT_filename, experiment, vairable  # col to plot i.e. HT_20
):
    HT_df = getHeadTwitchDf(HT_filename)

    data = HT_df[HT_df["experiment"] == experiment].rename(
        columns={vairable: "value"}
    )  # subselect experiment and set vairable col to 'value'

    order = data.sort_values(by="group_id", ascending=True).treatment.unique()
    palette = {
        treatment: color
        for treatment, color in data.groupby(by=["treatment", "color"]).groups.keys()
    }

    return data, order, palette




def buildHistogram(
    title,
    ylabel,
    data,
    order,
    hue=None,
    palette=None,
    swarm_hue=None,
    swarm_palette=None,
    significance_infos=None,  # x='treatment',y='value'
):
    # JASMINE: in what case would the x and y be variables? #REMI we need to talk about this func as it should be more general
    x = "treatment"
    y = "value"

    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.barplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        palette=palette,
        errorbar=("ci", 68),
        order=order,
        capsize=0.1,
        alpha=0.8,
        errcolor=".2",
        edgecolor=".2",
        dodge=False,
    )
    # #REMI so thiis for the outliers! I was trying to have this function work for my other histogram needs but i cant with this
    ax = sns.swarmplot(
        x=x,
        y=y,
        hue=swarm_hue or hue,
        palette=swarm_palette or palette,
        order=order,
        data=data,
        edgecolor="k",
        linewidth=1,
        linestyle="-",
        dodge=False,
        legend=True if swarm_palette else False,
    )

    if significance_infos:
        ax = labelStats(ax, data, x, y, order, significance_infos)

    ax.tick_params(labelsize=24)
    # ax.set_ylabel(ylabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xlabel(" ", fontsize=20)  # treatments
    ax.set_title(title, y=1.04, fontsize=34)  # '+/- 68%CI'
    sns.despine(left=False)
    return fig

def labelStats(ax, data, x, y, order, significance_infos):
    pairs, p_values = significance_infos
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
    annotator.configure(text_format="star", loc="inside", fontsize="xx-large")
    annotator.set_pvalues_and_annotate(p_values)

    return ax

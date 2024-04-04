from matplotlib import pyplot as plt
import numpy as np 
import seaborn as sns
from module.getters import getRawDf, getExperimentalInfo, subselectDf, getQuantitativeStats
from module.stats import processQuantitativeStats, updateQuantitativeStats, getPostHocTest
from module.utils import figure_cache
from statannotations.Annotator import Annotator



@figure_cache("head_twitch_histogram")
def singleHistogram(
    filename,
    experiment=None,
    vairable=None,
    outlier_test=None,
    p_value_threshold=None,
    from_scratch=None,
):
    '''
    To be called by user. Plots histogram for a single bevahior and timepoint - fetches data, does stats, plots. 
    input:
        filename (str) : 
        
    returns: 
        figure
    '''
    HT_data = getRawDf(filename)
    data, order, palette = buildsingleHistogramData(
        filename, experiment, vairable
    )

    title = vairable.replace("_", " at ") + " min"
    ylabel = "events / min"

    # REMI add stats calcultaion to input to histogram builder and outlier same as for quantativeHistograms()
    # the last quantitative test is coded to return the labels directly, thus the need for the bool
    (is_significant, significance_infos, test_results) = processQuantitativeStats(
        getExperimentalInfo(filename)[experiment], data, p_value_threshold
    )

    updateQuantitativeStats(
        filename,
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

@figure_cache("behavior_histogram")
def behaviorHistogram( filename, 
                      experiment=None, 
                      behavior=None, 
                      p_value_threshold = 0.05, 
                      from_scratch = True ):
    '''
    inpput:
        filename (str)      :
        experiment (str)    :         
        vairable (str)      :         col to plot i.e. 'HT_20' 
        
    returns: 
        fig ()              :         histogram for all timepoints for an experiment and behavior with corrisponding stats relative to treatment[0]
    '''

    raw_df = getRawDf(filename)
    plotting_df, timepoints = filterColumnsExperimentBehavior(raw_df, experiment, behavior)

    order = plotting_df.sort_values(by="group_id", ascending=True).treatment.unique()
    palette = {
        treatment: color
        for treatment, color in plotting_df.groupby(by=["treatment", "color"]).groups.keys()
    }
    title = f'{behavior} for {experiment}'
    ylabel = f'{behavior}/minute'

    #fetch stats 
    for time in timepoints: # time replaces region as it is the x-tixk rename to general after #TODO
        plotting_df_temp = plotting_df.rename(columns={behavior:'value'}).copy()
        plotting_df_temp = plotting_df_temp[plotting_df_temp['time']==time]
        (is_significant, significance_infos, test_results) = processQuantitativeStats(
            getExperimentalInfo(filename)[experiment], plotting_df_temp, p_value_threshold
        )

        updateQuantitativeStats(
            filename,
            [
                {
                    "data_type": behavior,
                    "experiment": experiment,
                    "compound": None,
                    "region": time,
                    **test_result,
                }
                for test_result in test_results
            ],
        )

    test = getPostHocTest(filename, experiment)
    quant_stats_df = subselectDf(
        getQuantitativeStats(filename),
        {
            "experiment": experiment,
            "data_type": behavior,
            "test": test,
        },
        )

    #plot 
    fig = buildHueHistogram(
        title,
        ylabel,
        plotting_df,
        timepoints,
        x='time',
        y=behavior,
        hue='treatment',
        palette=palette,
        hue_order=order,
        significance_infos=quant_stats_df,
    )
    return fig 



def filterColumnsExperimentBehavior(df, experiment, behavior):
    """
    Filter columns of a DataFrame for a single experiment and behavior, converts to long format for plotting.
    
    Args:
        df (pd.DataFrame)   :     long format with id_columns == ['mouse_id', 'group_id', 'treatment', 'color', 'experiment']
        experiment (str)    :     experiment identifier 
        behavior (str)      :     behavior identifier for cols behavior_time

    Returns:
        pd.DataFrame        :     subselected for single experiment and behavior columns with '_' split and last element taken 
    """
    id_columns = ['mouse_id', 'group_id', 'treatment', 'color', 'experiment']

    #create df 
    single_experiment_df = df[df['experiment'] == experiment]
    behavior_columns = [col for col in single_experiment_df.columns if behavior in col]
    behavior_df = single_experiment_df[id_columns + behavior_columns].copy()
    timepoints = [col.split('_')[-1] for col in behavior_columns] #remove behavior form col names
    behavior_df.columns = id_columns + timepoints

    #convert to long format 
    plotting_df = behavior_df.melt( id_vars = ['mouse_id', 'group_id', 'treatment', 'color', 'experiment'],
                                    value_vars = timepoints,
                                    var_name = 'time', 
                                    value_name = behavior )
    

    return plotting_df, timepoints


def buildsingleHistogramData( filename, experiment, vairable ):
    '''
    inpput:
        filename (str) :
        experiment (str) :
        vairable (str) :         col to plot i.e. 'HT_20' 
        
    returns: 
        data (pd.DataFrame) : 
        order () :
        palette (dict) :
    '''
    HT_df = getRawDf(filename)

    data = HT_df[HT_df["experiment"] == experiment].rename(
        columns={vairable: "value"}
    )  # subselect experiment and set vairable col to 'value'

    order = data.sort_values(by="group_id", ascending=True).treatment.unique()
    palette = {
        treatment: color
        for treatment, color in data.groupby(by=["treatment", "color"]).groups.keys()
    }

    return data, order, palette

def buildHueHistogram(
    title,
    ylabel,
    data,
    order,
    x=None,
    y=None,
    hue=None,
    palette=None,
    hue_order=None,
    significance_infos=None,
):
    '''
    Plots histogram of quanatative data for each treatment across region subset. #TODO need to make reversible 
    Args:
        title (string): 
        ylable (string):
        data (pd.DataFrame):                    subset of data - single compound - region subset - experiment
        order (list):                           list of strings corrisponding to x-ticks ['', '', ...]
        x (string):                             column in data of x values
        y (string):                             column in data of y values
        hue (string):                           column name for hue in data  i.e. 'treatment'
        palette (dict):                         dict mapping hue colors {'treatment1':'color1', ... }
        hue_order ( np.array(string) ):         string of each hue in order
        significance_infos (pd.DataFrame):      QuantativeStats for data * post hoc test *  optional pram

    Retuns:
        fig (figure):                           histograms/barplot 
    '''
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.barplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        palette=palette,
        errorbar=("ci", 68),
        errwidth=1,
        order=order,
        hue_order=hue_order,
        capsize=0.1,
        alpha=0.8,
        errcolor=".2",
        edgecolor=".2",
    )
    # Set the size of x and y ticks
    ax.tick_params(labelsize=16)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.yaxis.set_label_coords(-0.035, 0.5)
    ax.set_xlabel(" ", fontsize=20)  # remove x title
    ax.set_title(title, y=1.04, fontsize=34)
    ax.legend(loc="upper right")  # , bbox_to_anchor=(0.1, 1))
    plt.tight_layout()

    # #handel significance info #already filtered to post hoc (check higher test passes?) # single compound across regions
    
    if significance_infos is not None:
        comparison_hue = hue_order[0] # stats compare against 'vehicles' 
        #loop seaborn bars : xtick1, hue1 --> xtick2, hue1 --> xtick3, hue1   #seaborn v0.11.2
        for i, bar in enumerate(ax.patches):


            # Calculate the index of the group (region) and the hue for this bar
            group_index = i % len(order)
            hue_index = i // len(order)
            # Get the region and hue names based on their indices
            region = order[group_index]
            hue = hue_order[hue_index]
            
            if hue == comparison_hue:
                continue # do not plot stats on control/vehicle

            if region not in significance_infos['region'].unique():
                continue #continue if no stats for that region

            # Check if this hue is in a significant pair for this region
            significant_posthoc_pairs = significance_infos[significance_infos['region'] == region]['p_value'].values[0][0]  #  [ [(hue, hue), (hue,hue)] ,   [p_val, p_val] ]
            significant_posthoc_p_values = significance_infos[significance_infos['region'] == region]['p_value'].values[0][1]
            for pair, p_value in zip(significant_posthoc_pairs, significant_posthoc_p_values):
                if comparison_hue in pair and hue in pair:
                    ax.text(bar.get_x() + bar.get_width() / 2,  0.1, '*', ha='center', va='bottom', fontsize=16)
                    # ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, '*', ha='center', va='bottom', fontsize=16)


                    break  

    sns.despine(left=False)
    return fig



def buildHistogram(
    title,
    ylabel,
    data,
    order,
    hue=None,
    palette=None,
    swarm_hue=None,
    swarm_palette=None,
    significance_infos=None, 
):
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

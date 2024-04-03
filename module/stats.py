import pingouin as pg
import scipy
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
# from outliers import smirnov_grubbs as grubbs #not used and having circular import shit 
import matplotlib.pyplot as plt
from module.utils import select_params, maskDf, cache
from module.getters import getQuantitativeStats


def processQuantitativeStats(experiment_info, data, p_value_threshold):
    """_summary_

    Args:
        experiment_info (dict): experiment info mapping
        data (df): data for signle experiment
        p_value_threshold (float): threshold applied to all tests

    Returns:
        is_significant (bool), significance_infos (list, list): [treatment parings, pvalues]
    """
    multiple_factors = len(experiment_info["independant_vars"]) > 1
    multiple_treatments = len(experiment_info["groups"]) > 1
    paired = experiment_info["paired"]
    parametric = experiment_info["parametric"]

    tests = doQuantitativeStatLogic(
        multiple_factors, multiple_treatments, paired, parametric
    )
    test_results = []
    for test in tests:
        p_value, is_significant, stats_results = QUANTITATIVE_STAT_METHODS[test](
            data=data[data.value.notna()],
            independant_vars=experiment_info["independant_vars"],
            p_value_threshold=p_value_threshold,
        )
        print()
        print(test.upper(), "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT")
        print(stats_results)
        test_results.append(
            {
                "test": test,
                "is_significant": is_significant,
                "result": stats_results,
                "p_value_threshold": p_value_threshold,
                "p_value": p_value,
            }
        )

    return is_significant, p_value, test_results

def doQuantitativeStatLogic(multiple_factors, multiple_treatments, paired, parametric):
    return {
        (False, False, False, True): ["ttest"],
        (False, False, True, True): ["paired_ttest"],
        (False, True, False, True): ["one_way_anova", "tukey"],
        (False, True, True, True): ["repeated_measures_anova", "paired_ttest"],
        (True, True, False, True): ["two_way_anova", "one_way_anova", "tukey"],
    }[(multiple_factors, multiple_treatments, paired, parametric)]



def updateQuantitativeStats(filename, rows):
    quantitative_stats_df = getQuantitativeStats(filename)
    for row in rows:
        data_row = pd.DataFrame([row])
        unique_row = maskDf(
            quantitative_stats_df,
            {
                key: value
                for key, value in row.items()
                if key
                in [
                    "data_type",
                    "experiment",
                    "region",
                    "compound",
                    "test",
                    "p_value_threshold",
                ]
            },
        )
        if unique_row.any():
            quantitative_stats_df.loc[
                unique_row, ["is_significant", "p_value", "result"]
            ] = data_row[["is_significant", "p_value", "result"]]
        else:
            quantitative_stats_df = pd.concat([quantitative_stats_df, data_row])
    cache(filename, "quantitative_stats", quantitative_stats_df)
    print("QUANTITATIVE STATS UPDATED")


######## STATS FROM HPLC ######## 3/4/24

# The following functions are just here to be passed to the pd.corr() method, c and y are the two lists of values (columns) to be correlated
# This is a classic design pattern of which i forget the name again.
def isSignificant(stat_method_cb, pval_threshold=0.05):
    # As you can see here it will return a function. NOT CALL THE FUNCTION, but return it. The point here is to inject variables in the function that is returned.
    # when isSignificant(callback, pval_threshold) is called, it declare the anonymous function (lambda) passing the variables, and returns this declaration
    # this means that when the lambda function is called later on these will be 'harcoded' in the sense that they are no longer variables to be passed
    # Why? because if you look at the usage I pass it to pd.corr() to generate the mask in getPearsonCorrStats(). The thing is that pd.corr() calls the method it is given with (x, y) arguments
    # For the mask to be properly generated however, we also want to give a pval threshold to pass, as well as the statistical method that determines significance
    # But there is no way to pass this once you are in th pd.corr() context, so this is why you use this design pattern. It is meant to import variable into a context where they are not normally available
    return lambda x, y: stat_method_cb(x, y) >= pval_threshold


def getPearson(x, y):
    return scipy.stats.pearsonr(x, y)


def getPearsonR(x, y):
    return getPearson(x, y).statistic


def getPearsonPValue(x, y):
    return getPearson(x, y).pvalue

##CHECK ME JJB REMI TODO
def getSpearman(x, y):
    return scipy.stats.spearmanr(x,y)

def getSpearmanR(x, y):
    return getSpearman(x, y).correlation

def getSpearmanPValue(x, y):
    return getSpearman(x, y).pvalue

def getKendall(x, y):
    return scipy.stats.kendalltau(x, y)

def getKendallTau(x, y):
    return getKendall(x, y).correlation

def getKendallPValue(x, y):
    return getKendall(x, y).pvalue


@select_params
def getTukey(data, p_value_threshold):
    columns, *stats_data = pairwise_tukeyhsd(
        endog=data["value"], groups=data["treatment"], alpha=p_value_threshold
    )._results_table.data
    results = pd.DataFrame(stats_data, columns=columns)
    significance_infos = pd.DataFrame(
        list(
            results[results.reject].apply(
                lambda res: [(res.group1, res.group2), res["p-adj"]], axis=1
            )
        ),
        columns=["pairs", "p_values"],
    )
    return (
        [significance_infos.pairs.tolist(), significance_infos.p_values.tolist()],
        len(significance_infos) > 0,
        results,
    )


@select_params
def getOneWayAnova(data, p_value_threshold):
    F_value, p_value = scipy.stats.f_oneway(
        *[list(group_df["value"]) for treatment, group_df in data.groupby("treatment")]
    )
    # print(f'oneWAY_ANOVA F_value: {F_value}, p_value: {p_value}')
    return p_value, p_value <= p_value_threshold, pd.DataFrame(
        [[F_value, p_value]], columns=["F", "p_value"]
    )


@select_params
def getTwoWayAnova(data, independant_vars, p_value_threshold):
    data[independant_vars] = data.apply(
        lambda x: [var in x["treatment"] for var in independant_vars],
        axis=1,
        result_type="expand",
    )
    results = pg.anova(
        data=data,
        dv="value",
        between=independant_vars,
        detailed=True,
    ).round(3)
    return (results["p-unc"][2],
        isinstance(results["p-unc"][2], float)
        and results["p-unc"][2] < p_value_threshold,
        results,
    )


QUANTITATIVE_STAT_METHODS = {
    "two_way_anova": getTwoWayAnova,
    "one_way_anova": getOneWayAnova,
    "tukey": getTukey,
}


CORR_STAT_METHODS = {'pearson': [getPearson, getPearsonR, getPearsonPValue], 
                     'spearman': [getSpearman, getSpearmanR, getSpearmanPValue], 
                     'kendall': [getKendall, getKendallTau, getKendallPValue] }

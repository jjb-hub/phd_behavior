# from matplotlib import pyplot as plt
# import pandas as pd
# # from module.quantitative import processQuantitativeStats
# from getters import getRawDf, getExperimentalInfo, updateQuantitativeStats
# from histogram import buildsingleHistogramData, buildHistogram
# from module.utils import figure_cache
# from module.stats import processQuantitativeStats
# import seaborn as sns







# @figure_cache("head_twitch_histogram")
# def singleHistogram(
#     HT_filename,
#     experiment=None,
#     vairable=None,
#     outlier_test=None,
#     p_value_threshold=None,
#     from_scratch=None,
# ):
#     HT_data = getRawDf(HT_filename)
#     data, order, palette = buildsingleHistogramData(
#         HT_filename, experiment, vairable
#     )

#     title = vairable.replace("_", " at ") + " min"
#     ylabel = "events / min"

#     # REMI add stats calcultaion to input to histogram builder and outlier same as for quantativeHistograms()
#     # the last quantitative test is coded to return the labels directly, thus the need for the bool
#     (is_significant, significance_infos, test_results) = processQuantitativeStats(
#         getExperimentalInfo(HT_filename)[experiment], data, p_value_threshold
#     )

#     updateQuantitativeStats(
#         HT_filename,
#         [
#             {
#                 "data_type": "HT",
#                 "experiment": experiment,
#                 "compound": None,
#                 "region": None,
#                 **test_result,
#             }
#             for test_result in test_results
#         ],
#     )

#     fig = buildHistogram(
#         title,
#         ylabel,
#         data,
#         order,
#         hue='treatment',
#         palette=palette,
#         significance_infos=significance_infos if is_significant else None,
#     )
#     return fig

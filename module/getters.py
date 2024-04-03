import scipy
import os
import pandas as pd
import numpy as np
from module.utils import (
    flatten,
    isCached,
    getCache,
    cache,
    getJSON,
    dictToFilename,
    askYesorNo,
    subselectDf,
    maskDf,
    replaceall,
)
from module.constants import CACHE_DIR, INPUT_DIR


def getOrBuildDf(filename, df_identifier, builder_cb):
    filename_no_extension = filename.split(".")[0]
    # Check cache to avoid recalcuating from scratch if alreasy done
    if isCached(filename_no_extension, df_identifier):
        return getCache(filename_no_extension, df_identifier)
    # Build useing callback otherwise and cache result
    print(f'BUILDING "{df_identifier}"')
    df = builder_cb(filename)
    cache(filename_no_extension, df_identifier, df)
    return df


def getRawDf(filename):
    return getOrBuildDf(filename, "raw_df", buildRawDf)

def buildRawDf(filename):
    file_name, file_type = filename.split(".")
    filepath = f"{INPUT_DIR}/{filename}"
    if not os.path.isfile(filepath):
        raise Exception(f'FILE {filename} IS ABSENT IN "input/" DIRECTORY')
    if file_type == "xlsx":
        data = pd.read_excel(filepath, header=0)
    if file_type == "csv":
        data = pd.read_csv(filepath, header=0)
    data.columns = [replaceall(col, {"-": "_", " ": "_"}) for col in data.columns]
    return data


def applyTreatmentMapping(df, filename):
    filename = filename.split(".")[0]
    treatment_mapping_path = f"{CACHE_DIR}/{filename}/treatment_mapping.json"
    # Check treatment mapping is present
    if not os.path.isfile(treatment_mapping_path):
        raise Exception(
            "TREATMENT INFORMATION ABSENT, TO SAVE TREATMENT MAPPING RUN 'setTreatment(filename, treatment_mapping)'"
        )
    treatment_mapping = getJSON((treatment_mapping_path))
    # Get the future column names from one of the treatments
    new_columns = list(list(treatment_mapping.values())[0].keys())

#OLD causing settingcopy warning 
#     df[new_columns] = df.apply(
#     lambda x: treatment_mapping[str(int(x["group_id"]))].values(),
#     axis=1,
#     result_type="expand",
# )
    #new
    df.loc[:, new_columns] = df.apply(
    lambda x: pd.Series(treatment_mapping[str(int(x["group_id"]))]),
    axis=1
)
    
    # Get alll the values and assign to corresponding columns
    # Duplicate rows belonging to multiple experiment so that groupby can be done later
    return df.explode("experiments").rename(columns={"experiments": "experiment"})



def getHeadTwitchDf(filename):
    return getOrBuildDf(filename, "headtwitch_df", buildHeadTwitchDf)

def getTreatmentMapping(filename):
    return getMetadata(filename, "treatment_mapping")

def getExperimentalInfo(filename):
    return getMetadata(filename, "experimental_info")

def getMetadata(filename, metadata_type):
    return getJSON(f"{CACHE_DIR}/{filename.split('.')[0]}/{metadata_type}.json")

def getQuantitativeStats(filename):
    return getOrBuildDf(filename, "quantitative_stats", buildQuantitativeStatsDf)

######## BUILDERS 


def buildHeadTwitchDf(HT_filename):
    new_format_df = applyTreatmentMapping(getRawDf(HT_filename), HT_filename)
    return new_format_df

def buildQuantitativeStatsDf(filename):
    return pd.DataFrame(
        columns=[
            "data_type",
            "experiment",
            "region",
            "compound",
            "test",
            "p_value_threshold",
            "is_significant",
            "p_value",
            "result",
        ]
    )
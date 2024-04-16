#!/usr/bin/env python3
# Dataset column names
dataset_col = "Dataset"
uid_col = "uid"
age_col = "Age"
day_col = "Day"
response_col = "Response"
immage_col = "IMMAGE"
strain_col = "Strain"
accesion_col = "geo_accession"
strain_index_col = "strain_index"

# Dataset subsets
influenza_dicts = [
    {
        "Dataset": "GSE41080.SDY212",
        "Days": ["FC.HAI", "HAI.D28"],
        "Day0": "HAI.D0",
        "DayMFC": "HAI.MFC",
    },
    {
        "Dataset": "GSE48018.SDY1276",
        "Days": ["HAI.D28", "HAI.FC"],
        "Day0": "HAI.D0",
        "DayMFC": "HAI.MFC",
    },
    {
        "Dataset": "GSE59654.SDY404",
        "Days": ["HAI.D28", "FC.HAI"],
        "Day0": "HAI.D0",
        "DayMFC": "HAI.MFC",
    },
    {
        "Dataset": "GSE59743.SDY400",
        "Days": ["FC.HAI", "HAI.D28"],
        "Day0": "HAI.D0",
        "DayMFC": "HAI.MFC",
    },
    {"Dataset": "SDY67", "Days": ["FC.D28.HAI", "HAI.D28"], "Day0": "HAI.D0", "DayMFC": "HAI.MFC"},
    {"Dataset": "GSE59635.SDY63", "Days": ["FC", "HAI.D28"], "Day0": "HAI.D0", "DayMFC": "HAI.MFC"},
    # Has five subjects only
    {
        "Dataset": "GSE45735.SDY224",
        "Days": ["FC.HAI", "HAI.D21"],
        "Day0": "HAI.D0",
        "DayMFC": "HAI.MFC",
    },
    # Doesn't have a HAI measurement
    {"Dataset": "GSE47353.SDY80", "Days": ["D70.nAb", "FC.D70.nAb"], "Day0": "D0.nAb"},
    # Need to calculate MFC individually for these
    {
        "Dataset": "GSE48023.SDY1276",
        "Days": ["HAI.FC", "HAI.D28"],
        "Day0": "HAI.D0",
        "DayMFC": "generated.HAI.MFC",
    },
    {"Dataset": "SDY296", "Days": ["D28.HAI", "FC.HAI"], "Day0": "D0.HAI"},
]

dataset_day_dicts = [
    {"Dataset": "GSE125921.SDY1529", "Days": ["FC", "D84"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE13485.SDY1264", "Days": ["D60"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE13699.SDY1289", "Days": ["D28"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE169159", "Days": ["FC.D42", "D42"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE41080.SDY212", "Days": ["HAI.D28"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE45735.SDY224", "Days": ["HAI.D21"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {
        "Dataset": "GSE47353.SDY80",
        "Days": ["D70.nAb", "FC.D70.nAb"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {
        "Dataset": "GSE48018.SDY1276",
        "Days": ["nAb.D28", "nAb.FC"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {
        "Dataset": "GSE48023.SDY1276",
        "Days": ["nAb.FC", "nAb.D14"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {"Dataset": "GSE59635.SDY63", "Days": ["HAI.D28"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {
        "Dataset": "GSE59654.SDY180",
        "Days": ["FC.HAI", "HAI.D28"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {
        "Dataset": "GSE59654.SDY404",
        "Days": ["FC.HAI", "HAI.D28"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {
        "Dataset": "GSE59654.SDY520",
        "Days": ["FC.HAI", "HAI.D28"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {
        "Dataset": "GSE59743.SDY400",
        "Days": ["FC.HAI", "HAI.D28"],
        "Day0": "FIXME",
        "DayMFC": "FIXME",
    },
    {"Dataset": "GSE65834.SDY1328", "Days": ["D7", "FC"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE79396.SDY984", "Days": ["D28", "FC.D28"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "GSE82152.SDY1294", "Days": ["D28", "FC"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "SDY1325", "Days": ["FC.D28", "D28"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "SDY296", "Days": ["D28.nAb", "FC.nAb"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "SDY67", "Days": ["nAb.D28", "FC.D28.nAb"], "Day0": "FIXME", "DayMFC": "FIXME"},
    {"Dataset": "SDY89", "Days": ["D28"], "Day0": "FIXME", "DayMFC": "FIXME"},
]

exclude_datasets = ["GSE45735.SDY224", "GSE47353.SDY80"]


# Boolean flags
bAdjustMFC = False
bDiscardSeroprotected = True
bOlderOnly = False
bInfluenza = True

# Configurations
age_threshlod = 60
HAI_threshold = 40

if bAdjustMFC:
    exclude_datasets = ["GSE45735.SDY224", "GSE47353.SDY80", "GSE48023.SDY1276", "SDY296"]


age_restrict_str = f"_older-only" if bOlderOnly else ""
seroprotected_str = f"_discard_seroprotected" if bDiscardSeroprotected else ""

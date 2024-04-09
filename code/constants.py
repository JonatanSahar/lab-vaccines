#!/usr/bin/env python3
dataset_col = "Dataset"
uid_col = "uid"
age_col = "Age"
day_col = "Day"
response_col = "Response"
immage_col = "IMMAGE"
strain_col = "Strain"

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
    # Five subjects only
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
    {"Dataset": "GSE125921.SDY1529", "Days": ["FC", "D84"]},
    {"Dataset": "GSE13485.SDY1264", "Days": ["D60"]},
    {"Dataset": "GSE13699.SDY1289", "Days": ["D28"]},
    {"Dataset": "GSE169159", "Days": ["FC.D42", "D42"]},
    {"Dataset": "GSE41080.SDY212", "Days": ["HAI.D28"]},
    {"Dataset": "GSE45735.SDY224", "Days": ["HAI.D21"]},
    {"Dataset": "GSE47353.SDY80", "Days": ["D70.nAb", "FC.D70.nAb"]},
    {"Dataset": "GSE48018.SDY1276", "Days": ["nAb.D28", "nAb.FC"]},
    {"Dataset": "GSE48023.SDY1276", "Days": ["nAb.FC", "nAb.D14"]},
    {"Dataset": "GSE59635.SDY63", "Days": ["HAI.D28"]},
    {"Dataset": "GSE59654.SDY180", "Days": ["FC.HAI", "HAI.D28"]},
    {"Dataset": "GSE59654.SDY404", "Days": ["FC.HAI", "HAI.D28"]},
    {"Dataset": "GSE59654.SDY520", "Days": ["FC.HAI", "HAI.D28"]},
    {"Dataset": "GSE59743.SDY400", "Days": ["FC.HAI", "HAI.D28"]},
    {"Dataset": "GSE65834.SDY1328", "Days": ["D7", "FC"]},
    {"Dataset": "GSE79396.SDY984", "Days": ["D28", "FC.D28"]},
    {"Dataset": "GSE82152.SDY1294", "Days": ["D28", "FC"]},
    {"Dataset": "SDY1325", "Days": ["FC.D28", "D28"]},
    {"Dataset": "SDY296", "Days": ["D28.nAb", "FC.nAb"]},
    {"Dataset": "SDY67", "Days": ["nAb.D28", "FC.D28.nAb"]},
    {"Dataset": "SDY89", "Days": ["D28"]},
]

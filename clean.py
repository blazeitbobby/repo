import numpy as np
import pandas as pd


def clean_mi_pl_bs(file_path, sheet_name):
    """
    Designed for cleaning PL and BS.
    Handle multi-level header.
    """
    # Read the two header rows
    header_df = pd.read_excel(
        file_path, sheet_name=sheet_name, header=None, skiprows=4, nrows=2
    )

    # Multi-level header
    top_header_raw = header_df.iloc[0]
    bottom_header = header_df.iloc[1].tolist()

    top_header = top_header_raw.ffill().tolist()

    try:
        prior_year_index = bottom_header.index("Prior_Year")
        for i in range(prior_year_index, len(top_header)):
            top_header[i] = ""
    except ValueError:
        pass

    header_tuples = [
        (str(top) if pd.notna(top) else "", str(bot) if pd.notna(bot) else "")
        for top, bot in zip(top_header, bottom_header)
    ]

    if len(header_tuples) > 1:
        header_tuples[0] = ("", "Code")
        header_tuples[1] = ("", "Description")

    multi_header = pd.MultiIndex.from_tuples(header_tuples)

    # Read main data body using original number of columns from header
    num_cols = len(header_df.columns)
    data = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        skiprows=7,
        usecols=range(num_cols),
    )

    # Combine data and headers
    data.columns = multi_header

    # Drop fully empty columns and rows
    data.dropna(how="all", axis=1, inplace=True)
    data.dropna(how="all", axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def clean_mi_deferrals(file_path, sheet_name):

    df = pd.read_excel(file_path, sheet_name)
    df_clean = df.dropna(how="all").dropna(how="all", axis=1)

    df_clean.columns = df_clean.iloc[0]

    df_clean = df_clean[1:].reset_index(drop=True)

    return df_clean


def clean_gross_tp_walk(file_path, sheet_name):
    """
    Clean 'Gross TP walk'
    """

    title = pd.read_excel(
        file_path, sheet_name=sheet_name, header=None, usecols="A", nrows=1
    ).iloc[0, 0]

    walk_df = pd.read_excel(
        file_path, sheet_name=sheet_name, header=0, skiprows=1, usecols="A:F"
    )

    walk_df.dropna(axis=1, how="all", inplace=True)
    walk_df.rename(columns={"Unnamed: 0": title}, inplace=True)

    walk_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    walk_df.dropna(how="all", inplace=True)
    walk_df.reset_index(drop=True, inplace=True)

    return walk_df


def clean_unearned_premium_unexpired_risk(file_path, sheet_name):
    """
    Clean and combine three secondary tables from range H3:N20.
    """

    table_definitions = {
        "act_name": {"skiprows": 2, "nrows": 5, "usecols": "H:N"},
        "unearned_ulrs": {"skiprows": 8, "nrows": 5, "usecols": "H:N"},
        "unexpired_claims": {"skiprows": 14, "nrows": 6, "usecols": "H:N"},
    }

    base_headers = pd.read_excel(
        file_path, sheet_name=sheet_name, header=0, skiprows=2, nrows=1, usecols="H:N"
    ).columns.tolist()

    base_headers[1] = "Item"
    all_tables = []

    for name, region in table_definitions.items():
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=region["skiprows"] + 1,
            nrows=region["nrows"] - 1,
            usecols=region["usecols"],
        )
        df.columns = base_headers

        source_name = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=region["skiprows"],
            nrows=1,
            usecols=region["usecols"],
        ).iloc[0, 1]

        df.insert(0, "Source", source_name)
        all_tables.append(df)

    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df.dropna(how="all", axis=1, inplace=True)
    combined_df.dropna(how="all", axis=0, inplace=True)

    return combined_df


def collapse_header(header_list):
    """Collapse multiple header rows into a single header for S2 Balance Sheet and SCR."""
    transposed_headers = list(zip(*header_list))
    final_headers = []
    for col_parts in transposed_headers:
        clean_parts = [
            str(p) for p in col_parts if pd.notna(p) and "Unnamed" not in str(p)
        ]
        final_headers.append(" - ".join(clean_parts))
    return final_headers


def fill_merged_cells_in_groups(df):
    """Forward-fill the first column to handle merged cells."""
    first_col = df.columns[0]
    other_cols = df.columns[1:]
    is_separator = df[other_cols].isnull().all(axis=1)

    df["group"] = is_separator.cumsum()

    df[first_col] = df.groupby("group")[first_col].ffill()
    df.drop(columns="group", inplace=True)
    return df


def clean_s2_summary_sheet(file_path, sheet_name):
    """
    Clean S2 Balance Sheet, SCR, NL P&R risk calc, OP risk calc.
    """

    header_block_df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        skiprows=1,
        nrows=2,
        usecols="B:F",
    )
    shared_header_F = collapse_header(header_block_df.values.tolist())[-1]

    regions = {
        "s2_balance_sheet": {"range": "A2:F41", "header_rows": 2},
        "scr": {"range": "A43:F78", "header_rows": 3},
        "nl_pr_risk": {"range": "A80:F95", "header_rows": 1},
        "op_risk": {"range": "A98:F107", "header_rows": 1},
    }

    cleaned_tables = {}

    for name, region in regions.items():

        start_row = int(region["range"].split(":")[0][1:])
        end_row = int(region["range"].split(":")[1][1:])

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            usecols="B:F",
            skiprows=start_row - 1,
            nrows=(end_row - start_row + 1),
        )

        table_title = df.iloc[0, 0]

        if name in ["s2_balance_sheet", "scr"]:
            header_list = [list(df.iloc[i]) for i in range(region["header_rows"])]
            df.columns = collapse_header(header_list)
            df = df.iloc[region["header_rows"] :].reset_index(drop=True)
            df = fill_merged_cells_in_groups(df)
            df.rename(columns={df.columns[-1]: shared_header_F}, inplace=True)
        else:
            first_col_header = df.iloc[0, 0]
            num_cols = df.shape[1]

            new_headers = [first_col_header]
            new_headers.extend([""] * (num_cols - 2))
            new_headers.append(shared_header_F)

            df.columns = new_headers
            df = df.iloc[region["header_rows"] :].reset_index(drop=True)

        df.dropna(how="all", inplace=True)
        df.dropna(how="all", axis=1, inplace=True)

        df.name = table_title
        cleaned_tables[name] = df

    return cleaned_tables

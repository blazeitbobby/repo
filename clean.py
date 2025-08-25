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

    # Scaling

    # Read the full sheet into a temporary DataFrame to search for the indicator
    full_sheet_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Check all cells for the presence of a scaling indicator (e.g., '000s, £'000)
    scaling_indicator_found = (
        full_sheet_df.astype(str)
        .apply(lambda x: x.str.contains("'000|000s", case=False))
        .any()
        .any()
    )

    if scaling_indicator_found:

        numeric_cols = data.select_dtypes(include=np.number).columns

        # Determine rows to scale based on the sheet name
        if sheet_name == "PL":
            rows_to_scale = data.index[:-6]
        elif sheet_name == "BS":
            rows_to_scale = data.index[:-1]
        else:
            # Default for any other sheets is to scale all rows
            rows_to_scale = data.index

        # Apply scaling
        if len(rows_to_scale) > 0:
            data.loc[rows_to_scale, numeric_cols] = (
                data.loc[rows_to_scale, numeric_cols] * 1000
            )
        else:
            print(f"Found 0 rows to scale for {sheet_name}. Skipping multiplication.")
    else:
        print(
            f"No scaling indicator found in sheet '{sheet_name}'. Data will not be scaled."
        )

    return data


def clean_mi_deferrals(file_path, sheet_name):
    """
    Clean 'Deferrals' sheet of MI pack.
    """

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

    shared_headers = collapse_header(header_block_df.values.tolist())

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
            df.rename(columns={df.columns[-1]: shared_headers[-1]}, inplace=True)
        else:
            first_col_header = df.iloc[0, 0]

            new_headers = [first_col_header] + shared_headers[1:]

            df.columns = new_headers
            df = df.iloc[region["header_rows"] :].reset_index(drop=True)

        df.dropna(how="all", inplace=True)
        df.dropna(how="all", axis=1, inplace=True)

        df.name = table_title
        cleaned_tables[name] = df

    return cleaned_tables


def clean_version_control_solvency_balance_sheet(file_path, skip_rows, n_rows, element):
    """
    Process Main SOLVENCY BALANCE SHEET
    """

    df_headers = pd.read_excel(
        file_path, header=None, skiprows=3, nrows=3, usecols=range(0, 21)
    )

    # Create combined header
    header_part1 = df_headers.iloc[0].fillna("")
    header_part2 = df_headers.iloc[1].fillna("")
    header_part3 = df_headers.iloc[2].fillna("")

    new_header = [
        f"{h1} {h2} {h3}".strip()
        for h1, h2, h3 in zip(header_part1, header_part2, header_part3)
    ]

    df = pd.read_excel(
        file_path, header=None, skiprows=skip_rows, nrows=n_rows, usecols=range(0, 21)
    )

    df.columns = new_header

    df = df.dropna(how="all").dropna(how="all", axis=1)

    # Forward fill in col A
    df.iloc[:, 0] = df.iloc[:, 0].ffill()

    # Keep "Col C BS GAAP BS Jun-25" and "Col N BS SII BS Jun-25"
    df_keep = df.iloc[:, [0, 1, 9, -1]]

    df_filtered = df_keep[df_keep.count(axis=1) != 1]  # Remove rows wih all but one NaN

    # Rename first column to `element`
    old_name = df_filtered.columns[0]
    df_filtered = df_filtered.rename(columns={old_name: element})

    return df_filtered


def clean_version_control_scr_review(file_path):
    """
    Process SCR Review
    """

    df = pd.read_excel(
        file_path, header=None, skiprows=124, nrows=27, usecols=range(0, 22)
    )

    df = df.dropna(how="all").dropna(how="all", axis=1)

    # Create combined header
    header_part1 = df.iloc[0].fillna("")
    header_part2 = df.iloc[1].fillna("")
    new_header = [f"{h1} {h2}".strip() for h1, h2 in zip(header_part1, header_part2)]

    # Apply header
    df = df[2:]
    df.columns = new_header

    # Rename the final column
    df = df.rename(columns={df.columns[-1]: "Comments"})

    return df.reset_index(drop=True)


def clean_version_control_s2_gaap(file_path, skip_rows):
    """
    Process GAAP to SII and SII to GAAP
    """

    df = pd.read_excel(
        file_path, header=0, skiprows=skip_rows, nrows=23, usecols=range(23, 27)
    )

    df = df.dropna(how="all").dropna(how="all", axis=1)

    return df


def clean_version_control_s2_cap_prov_summary(file_path, skip_rows):
    """
    Process Capital Position Summary and Actual Solvency Position vs. Planned Solvency Position
    """

    df = pd.read_excel(
        file_path, header=0, skiprows=skip_rows, nrows=11, usecols=range(27, 31)
    )

    df = df.dropna(how="all").dropna(how="all", axis=1)

    numeric_cols = df.select_dtypes(include=np.number).columns

    rows_to_exclude = df.index[[-1, -7]]

    # Get the indices of the rows to scale by dropping the excluded ones
    rows_to_scale = df.index.drop(rows_to_exclude)

    # Scale the numeric values in the selected rows
    df.loc[rows_to_scale, numeric_cols] = df.loc[rows_to_scale, numeric_cols] * 1000

    return df

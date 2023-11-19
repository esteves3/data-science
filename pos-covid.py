from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart
from pandas import read_csv, DataFrame
from pandas import Series, to_numeric, to_datetime
from matplotlib.pyplot import figure, savefig, show, xticks,yticks
from dslabs_functions import plot_bar_chart, get_variable_types, dummify
from numpy import ndarray
from pandas import DataFrame, read_csv, concat
from sklearn.preprocessing import OneHotEncoder
from seaborn import heatmap

def dimensionality():
    filename = "datasets/class_pos_covid.csv"
    file_tag = "class_pos_covid"
    data: DataFrame = read_csv(filename, na_values="")
    figure(figsize=(4, 2))
    values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
    plot_bar_chart(
        list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
    )
    savefig(f"images/{file_tag}_records_variables.png")

    mv: dict[str, int] = {}
    for var in data.columns:
        nr: int = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    figure(figsize=(10,4))
    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values",
    )
    savefig(f"images/{file_tag}_mv.png")

    variable_types: dict[str, list] = get_variable_types(data)
    print(variable_types)
    counts: dict[str, int] = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])

    figure(figsize=(4, 4))
    plot_bar_chart(
        list(counts.keys()), list(counts.values()), title="Nr of variables per type"
    )
    savefig(f"images/{file_tag}_variable_types.png")

def sparsity():
    filename = "datasets/class_pos_covid.csv"
    file_tag = "class_pos_covid"
    data: DataFrame = read_csv(filename, na_values="")
    data = data.dropna()
    vars: list = data.columns.to_list()
    if [] != vars:
        target = "CovidPos"

        n: int = len(vars) - 1
        fig: Figure
        axs: ndarray
        fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
        for i in range(len(vars)):
            var1: str = vars[i]
            for j in range(i + 1, len(vars)):
                var2: str = vars[j]
                plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])
        savefig(f"images/{file_tag}_sparsity_study.png")
    else:
        print("Sparsity class: there are no variables.")

def sparsity_per_class():
    filename = "datasets/class_pos_covid.csv"
    file_tag = "class_pos_covid"
    data: DataFrame = read_csv(filename, na_values="")
    data = data.dropna()
    vars: list = data.columns.to_list()
    if [] != vars:
        target = "CovidPos"

        n: int = len(vars) - 1
        fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
        for i in range(len(vars)):
            var1: str = vars[i]
            for j in range(i + 1, len(vars)):
                var2: str = vars[j]
                plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
        savefig(f"images/{file_tag}_sparsity_per_class_study.png")
    else:
        print("Sparsity per class: there are no variables.")

def dummify2(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=bool, drop='if_binary')
    trans = enc.fit_transform(df[vars_to_dummify])
    print(trans)

    new_vars = enc.get_feature_names_out(vars_to_dummify)
    dummy = DataFrame(trans, columns=new_vars, index=df.index)

    final_df = concat([df[other_vars], dummy], axis=1)
    return (final_df,new_vars)

def correlation():
    filename = "datasets/class_pos_covid.csv"
    file_tag = "class_pos_covid"
    data: DataFrame = read_csv(
        "datasets/class_pos_covid.csv", na_values="", parse_dates=True, dayfirst=True
    )

    yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
    sex: dict[str, int] = {"Male": 0, "Female": 1}

    encoding: dict[str, dict[str, int]] = {
        'Sex':sex,
        'PhysicalActivities':yes_no,
        'HadHeartAttack':yes_no,
        'HadAngina':yes_no,
        'HadStroke':yes_no,
        'HadAsthma':yes_no,
        'HadSkinCancer':yes_no,
        'HadCOPD':yes_no,
        'HadDepressiveDisorder':yes_no,
        'HadKidneyDisease':yes_no,
        'HadArthritis':yes_no,
        'DeafOrHardOfHearing':yes_no,
        'BlindOrVisionDifficulty':yes_no,
        'DifficultyConcentrating':yes_no,
        'DifficultyWalking':yes_no,
        'DifficultyDressingBathing':yes_no,
        'DifficultyErrands':yes_no,
        'ChestScan':yes_no,
        'AlcoholDrinkers':yes_no,
        'HIVTesting':yes_no,
        'FluVaxLast12':yes_no,
        'PneumoVaxEver':yes_no,
        'HighRiskLastYear':yes_no,
        'CovidPos':yes_no,
    }
    df: DataFrame = data.replace(encoding, inplace=False)
    vars: list[str] =['State', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap']
    df2: DataFrame = dummify2(df, vars)[0]
    b_vars = dummify2(df, vars)[1]
    df2[b_vars] = df2[b_vars].astype(int)


    variables_types: dict[str, list] = get_variable_types(df2)
    print(variables_types)
    numeric: list[str] = variables_types["numeric"]
    corr_mtx: DataFrame = df2.corr().abs()

    figure(figsize=(25, 25)) 
    heatmap(
        abs(corr_mtx),
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    xticks(fontsize='x-small')
    yticks(fontsize='x-small')
    savefig(f"images/{file_tag}_correlation_analysis.png")
from dslabs_functions import get_variable_types
from seaborn import heatmap
from dslabs_functions import HEIGHT, plot_multi_scatters_chart
from matplotlib.pyplot import figure, subplots, savefig, show, gcf
from dslabs_functions import plot_bar_chart
from dslabs_functions import set_chart_labels
from dslabs_functions import define_grid, HEIGHT
from matplotlib.figure import Figure
from numpy import ndarray
from dslabs_functions import *
from pandas import read_csv, DataFrame
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm
from matplotlib.axes import Axes
from dslabs_functions import plot_multiline_chart
import math

# %%
filename = "datasets/class_credit_score.csv"
file_tag = "credit_score"
data: DataFrame = read_csv(filename, na_values="", index_col="ID")

# %%
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.max_colwidth", 200)

# %%
# Remove non-digits from age column
data['Age'] = data['Age'].str.replace(r'[^0-9]+', '', regex=True)

# Drop name column
data = data.drop(columns=['Name'])

# Leave only area code for SSN
data['SSN'] = data['SSN'].str.slice(stop=3)
data = data.rename(columns = {'SSN': 'SSN_Area_Code'})


# %%
def process_loan_type_entry(entry):
    if not isinstance(entry, float):
        loan_types_split = []
        type_list = entry.replace(' and ', ' ')
        type_list = type_list.split(', ')
        for loan_type in type_list:
            loan_types_split.append('Loan_Type_' + loan_type.strip().replace(' ', '_').replace('-', '_'))
        return loan_types_split
    return []

# Split loan types and reformat the strings

loan_copy = data['Type_of_Loan']
no_nans = data.dropna()
loan_values = no_nans['Type_of_Loan'].unique()

loan_types = []
for entry in loan_values:
    loan_types += process_loan_type_entry(entry)

loan_types_columns = set(loan_types)
loan_types_columns = list(loan_types_columns)
print(loan_types_columns)

def columns_count_occurrences(column_names, list_to_count):

    column_values = dict.fromkeys(column_names, 0)
    for item in list_to_count:
        column_values[item] += 1
    return column_values


data[loan_types_columns] = data.apply(lambda row: columns_count_occurrences(loan_types_columns, process_loan_type_entry(row['Type_of_Loan'])), axis='columns', result_type='expand')

no_nans.head(7)

no_nans[['NumofLoan', 'Type_of_Loan']].head(15)


import re

def convert_age_to_months(age):
    if isinstance(age, str):
        list_of_numbers = re.findall(r'\b\d+\b', age)
        if (len(list_of_numbers) != 2):
            print(list_of_numbers)
            raise Exception('Incorrect age input')
        years, months = int(list_of_numbers[0]), int(list_of_numbers[1])
        total_months = years * 12 + months
        return total_months
    return 0

data['Credit_History_Age_Months'] = data.apply(lambda row: convert_age_to_months(row['Credit_History_Age']), axis='columns', result_type='expand')
data = data.drop(columns=["Credit_History_Age"])


payment_behaviour_enc: dict[str, int] = {"High_spent_Small_value_payments": 10, "Low_spent_Large_value_payments": 2, 
                                         "Low_spent_Medium_value_payments": 1, "Low_spent_Small_value_payments": 0,
                                         "High_spent_Medium_value_payments": 11, "High_spent_Large_value_payments": 12}


credit_mix_enc: dict[str, int] = {"Good": 2, "Standard": 1, "Bad": 0}

credit_score_enc: dict[str, int] = {"Good": 1, "Poor": 0}

payment_min_amount_enc: dict[str, int] = {"Yes": 2, "NM": 1, "No": 0}

month_val: dict[str, float] = {
    "January": 0,
    "February": pi / 4,
    "March": 2 * pi / 4,
    "April": 3 * pi / 4,
    "May": pi,
    "June": - 3 * pi / 4,
    "July": - 2 * pi / 4,
    "August": - pi / 4
}

print(month_val)

def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    other_vars: list[str] = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype="bool", drop="if_binary"
    )
    trans: ndarray = enc.fit_transform(df[vars_to_dummify])

    new_vars: ndarray = enc.get_feature_names_out(vars_to_dummify)
    dummy = DataFrame(trans, columns=new_vars, index=df.index)

    final_df: DataFrame = concat([df[other_vars], dummy], axis=1)
    return final_df

encoding: dict[str, dict[str, int]] = {
    "Payment_Behaviour": payment_behaviour_enc,
    "CreditMix": credit_mix_enc,
    "Month": month_val,
    "Credit_Score": credit_score_enc,
    "Payment_of_Min_Amount": payment_min_amount_enc
}

data = data.replace(encoding, inplace=False)

vars = ["Occupation"]
data = dummify(data, vars)

data['Customer_ID'] = data.apply(lambda row: int(row['Customer_ID'].replace('CUS_', ''), 16), axis='columns', result_type='expand')



def encode_cyclic_variables(data: DataFrame, vars: list[str]):
    _data = data
    for v in vars:
        x_max: float = max(data[v])
        _data[v + "_sin"] = data[v].apply(lambda x: round(sin(2 * pi * x / x_max), 5))
        _data[v + "_cos"] = data[v].apply(lambda x: round(cos(2 * pi * x / x_max), 5))
    return _data


data = encode_cyclic_variables(data, ["Month"])

print(data)
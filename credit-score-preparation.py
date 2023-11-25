# %% [markdown]
# ## Setup

# %%
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

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Cleaning

# %%
# Remove non-digits from age column
data['Age'] = data['Age'].str.replace(r'[^0-9]+', '', regex=True)

# Drop name column
data = data.drop(columns=['Name'])

# Leave only area code for SSN
data['SSN'] = data['SSN'].str.slice(stop=3)
data = data.rename(columns = {'SSN': 'SSN_Area_Code'})

data

# %%
def process_loan_type_entry(entry):
    loan_types_split = []
    type_list = entry.replace(' and ', ' ')
    type_list = type_list.split(', ')
    for loan_type in type_list:
        loan_types_split.append('Loan_Type_' + loan_type.strip().replace(' ', '_').replace('-', '_'))
    return loan_types_split

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


# Create columns and add to dataframe

def columns_count_occurrences(column_names, list_to_count):
    column_values = dict.fromkeys(column_names, 0)
    for item in list_to_count:
        column_values[item] += 1
    return column_values


no_nans[loan_types_columns] = no_nans.apply(lambda row: columns_count_occurrences(loan_types_columns, process_loan_type_entry(row['Type_of_Loan'])), axis='columns', result_type='expand')

no_nans.head(7)

# %%
# Can we drop "num of loan"?

no_nans[['NumofLoan', 'Type_of_Loan']].head(15)

# %%
# Convert credit history to months

import re

def convert_age_to_months(age):
    list_of_numbers = re.findall(r'\b\d+\b', age)
    if (len(list_of_numbers) != 2):
        print(list_of_numbers)
        raise Exception('Incorrect age input')
    years, months = int(list_of_numbers[0]), int(list_of_numbers[1])
    total_months = years * 12 + months
    return total_months

no_nans['Credit_History_Age_Months'] = no_nans.apply(lambda row: convert_age_to_months(row['Credit_History_Age']), axis='columns', result_type='expand')

no_nans[['Credit_History_Age', 'Credit_History_Age_Months']].head()

# %%
# Split payment behaviour and one-hot-encode

print(data['Payment_Behaviour'].unique())

# %% [markdown]
# ### Variables encoding
# 
# The list of variables under each one of the transformations, shall be presented. If not applied explain the reason for that, based on data characteristics.

# %%
print(get_variable_types(data)['symbolic'])
for var in get_variable_types(data)['symbolic']:
    print(var + ':')
    print(data[var].describe())
    print(data[var].unique())
    print()

# %% [markdown]
# ### Missing value imputation

# %%


# %% [markdown]
# ### Outliers treatment

# %%


# %% [markdown]
# ### Scaling

# %%


# %% [markdown]
# ### Balancing

# %%


# %% [markdown]
# ### Feature selection

# %%




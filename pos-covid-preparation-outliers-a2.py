#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series
from dslabs_functions import determine_outlier_thresholds_for_var, get_variable_types, encode_cyclic_variables, dummify, mvi_by_filling, evaluate_approach, plot_multibar_chart
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


# In[3]:


data: DataFrame = read_csv("datasets/class_pos_covid.csv", index_col=None, na_values=None)
vars: dict[str, list] = get_variable_types(data)

binaries: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1, "Female":0, "Male":1}
encoding: dict[str, dict[str, int]] = {}
for bin_col in vars["binary"]:
	encoding[bin_col] = binaries
df: DataFrame = data.replace(encoding, inplace=False)
df.head()


# Dorian
# state -- population
# general health -- ordinal liear
# last checkup -- the start of the interval like <1y == 0, >5y == 5
# smoker == ecigs -- 00 == nver, 01 == former, 11 == smokes but 
# Ruben
# race -- dumification
# age -- ordinal liear
# tetanos -- 00 == no shot, 10 == yes dont know , 11 == yes, knows but not tdap, 12 == yes and is tdap
# had diabites -- 00 == no, 10 == yes, 01 == no but is pre diabetic, 11 == yes and before pregancy
# teeth -- mid point of interval, 0 teeth = 0, 1-5 == 3, 6+ == 18, all == 32

# In[4]:


state : dict[str, int] = {"Virgin Islands": 105870,"Guam": 170534,"Wyoming": 578803,"Vermont": 	645570,
    "District of Columbia": 670050,"Alaska": 732673,"North Dakota": 774948,"South Dakota": 895376,"Delaware": 1003384,
    "Rhode Island": 1095610,"Montana" : 1104271,"Maine": 1372247,"New Hampshire": 1388992,"Hawaii": 1441553,
    "West Virginia": 1782959,"Idaho": 1900923,"Nebraska": 1963692,"New Mexico": 2115877,"Kansas": 2934582,
    "Mississippi": 2949965,"Arkansas": 3025891,"Nevada": 3143991,"Iowa": 3193079,"Puerto Rico": 3263584,
    "Utah": 3337975,"Connecticut": 3605597,"Oklahoma": 3986639,"Oregon": 4246155,"Kentucky": 4509394,
    "Louisiana": 4624047,"Alabama": 5039877,"South Carolina": 5190705,"Minnesota": 5707390,"Colorado": 5812069,
    "Wisconsin": 5895908,"Maryland": 6165129,"Missouri": 6168187,"Indiana": 6805985,"Tennessee": 6975218,
    "Massachusetts": 6984723,"Arizona": 7276316,"Washington": 7738692,"Virginia": 8642274,"New Jersey": 9267130,
    "Michigan" : 10050811,"North Carolina": 10551162,"Georgia": 10799566,"Ohio": 11780017,"Illinois": 12671469,
    "Pennsylvania": 12964056,"New York": 19835913,"Florida": 21781128,"Texas": 29527941,"California": 39237836}

health : dict[str,int] = {"Poor": 0,"Fair": 1,"Good": 2,"Very good": 3,"Excellent": 4}

last : dict[str,int] = {"Within past year (anytime less than 12 months ago)" : 0, 
    "Within past 2 years (1 year but less than 2 years ago)" : 1, 
    'Within past 5 years (2 years but less than 5 years ago)': 2,
    '5 or more years ago': 5}

smoke : dict[str,int] = {"Never smoked": 0,"Former smoker": 1,
    "Current smoker - now smokes some days": 3,
    "Current smoker - now smokes every day": 7}

ecig : dict[str,int] = {"Never used e-cigarettes in my entire life": 0,
    "Not at all (right now)": 1,
    "Use them some days": 3,
    "Use them every day": 7}

encoding['State'] = state
encoding['GeneralHealth'] = health
encoding['LastCheckupTime'] = last
encoding['SmokerStatus'] = smoke
encoding['ECigaretteUsage'] = ecig


df: DataFrame = data.replace(encoding, inplace=False)
df.head()


# # Coding RaceEthnicityCategory
# Applied dummification

# In[5]:


df = dummify(df, ["RaceEthnicityCategory"])
df.head()


# # Encoding AgeCategory
# 
# just ordinal linear encoding nothing special

# In[6]:


AgeCategory: dict[str, int] = {"Age 18 to 24":0, "Age 80 or older":12}
for i in range(11):
	AgeCategory[f'Age {i*5 + 25} to {i*5 + 29}'] = i+1
AgeCategory
encoding: dict[str, dict[str, int]] = {"AgeCategory":AgeCategory}
df: DataFrame = df.replace(encoding, inplace=False)
df.head()


# # Encoding TetanusLast10Tdap
# 
# tetanos -- 00 == no shot, 10 == yes dont know , 11 == yes, knows but not tdap, 12 == yes and is tdap

# In[7]:


TetanusLast10Tdap: dict[str, int] = {
	"Yes, received tetanus shot but not sure what type":2, 
	"No, did not receive any tetanus shot in the past 10 years":0, 
	"Yes, received Tdap":7, 
	"Yes, received tetanus shot, but not Tdap":3
}
encoding: dict[str, dict[str, int]] = {"TetanusLast10Tdap":TetanusLast10Tdap}
df: DataFrame = df.replace(encoding, inplace=False)
df.head()


# # Encoding HadDiabetes
# had diabites -- 00 == no, 10 == yes, 01 == no but is pre diabetic, 11 == yes and before pregancy

# In[8]:


HadDiabetes: dict[str, int] = {
	"No, pre-diabetes or borderline diabetes":1, 
	"Yes, but only during pregnancy (female)":3, 
	"Yes":2, 
	"No":0
}
encoding: dict[str, dict[str, int]] = {"HadDiabetes":HadDiabetes}
df: DataFrame = df.replace(encoding, inplace=False)
df.head()


# # Encoding RemovedTeeth
# 
# encode with the mid point of the interval
# 
# i.e. 0 teeth == 0, 1-5 == 2, 6+ == 18, all == 32

# In[9]:


RemovedTeeth: dict[str, int] = {"None of them":0, "1 to 5":2, "6 or more, but not all":18, "All":32}
encoding: dict[str, dict[str, int]] = {"RemovedTeeth":RemovedTeeth}
df: DataFrame = df.replace(encoding, inplace=False)
df.head()


# # Missing Value Imputation

# ## Approach 1
# 
# remove all records with mv

# In[10]:


df_mvi_a1: DataFrame = df.dropna(how="any", inplace=False)
df_mvi_a1 = df_mvi_a1.drop(df_mvi_a1[df_mvi_a1["RaceEthnicityCategory_nan"] == True].index)
df_mvi_a1 = df_mvi_a1.drop(columns=["RaceEthnicityCategory_nan"])
df_mvi_a1


# In[10]:


target = "CovidPos"
file_tag = "class_pos_covid_MVI_A1"

x = df_mvi_a1.drop(columns=[target])
y = df_mvi_a1[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test,y_test], axis=1)

plt.figure()
eval: dict[str, list] = evaluate_approach(df_train, df_test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
plt.savefig(f"images/{file_tag}_eval.png")
plt.show()


# ## Approach 2
# dont remove every record with missing values

# In[11]:


df_mvi_a2 = mvi_by_filling(df, strategy="frequent")
value_counnts = data["RaceEthnicityCategory"].value_counts()
most_occurences_value = f"RaceEthnicityCategory_{value_counnts.idxmax()}"

df_mvi_a2.loc[df_mvi_a2["RaceEthnicityCategory_nan"] == True, most_occurences_value] = True

df_mvi_a2 = df_mvi_a2.drop(columns=["RaceEthnicityCategory_nan"])
df_mvi_a2


# In[12]:


target = "CovidPos"
file_tag = "class_pos_covid_MVI_A2"
train, test = train_test_split(df_mvi_a2, test_size=0.2)

plt.figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
plt.savefig(f"images/{file_tag}_eval.png")
plt.show()


# ## Selecting approach

# In[12]:


df = df_mvi_a2
df


# # Outliers

# ## Approach 1

# Replace outliers with average except for BMI and WeightInKilograms

# In[13]:


numeric_vars: list[str] = get_variable_types(data)["numeric"]
exclude_vars: list[str] = ["BMI", "WeightInKilograms"]
numeric_vars_to_process: list[str] = [var for var in numeric_vars if var not in exclude_vars]

if [] != numeric_vars_to_process:
    df_outliers_a1: DataFrame = df.copy(deep=True)
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars_to_process:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        median: float = df[var].median()
        df_outliers_a1[var] =  df_outliers_a1[var].apply(lambda x: median if x > top or x < bottom else x)
    # df2.to_csv(f"data/{file_tag}_replacing_outliers.csv", index=True)
    print("Data after replacing outliers:", df_outliers_a1.shape)
else:
    print("There are no numeric variables")


# In[48]:


target = "CovidPos"
file_tag = "class_pos_covid_outliers_A1"
train, test = train_test_split(df_outliers_a1, test_size=0.2)

plt.figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
plt.savefig(f"images/{file_tag}_eval.png")
plt.show()


# # Scaling

# ## Approach 1
# 
# Using StandardScaler

# In[14]:


## Approach 1
target = "CovidPos"
vars: list[str] = df_outliers_a1.columns.to_list()
print(vars)
target_data: Series = df_outliers_a1.pop(target)
print(target_data)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    df_outliers_a1
)
df_zscore = DataFrame(transf.transform(df_outliers_a1), index=df_outliers_a1.index)
vars.remove(target)
df_zscore.columns = vars
df_zscore[target] = target_data
print(df_zscore.head())


# In[15]:


target = "CovidPos"
file_tag = "class_pos_covid_scaling_A1"
train, test = train_test_split(df_zscore, test_size=0.2)

plt.figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
plt.savefig(f"images/{file_tag}_eval.png")
plt.show()


# # Balancing

# ## Approach 1
# 
# Undersampling

# In[68]:


df_positives: Series = df[df[target] == 1]
df_negatives: Series = df[df[target] == 0]

df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = pd.concat([df_positives, df_neg_sample], axis=0)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")


# In[71]:


evaluate_approach_and_graph(df_under, file_tag="class_pos_covid_Balancing_A1")


# ## Approach 2
# 
# Oversampling by replication

# In[69]:


df_positives: Series = df[df[target] == 1]
df_negatives: Series = df[df[target] == 0]
df_pos_sample: DataFrame = DataFrame(
    df_positives.sample(len(df_negatives), replace=True)
)
df_over: DataFrame = pd.concat([df_pos_sample, df_negatives], axis=0)

print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(df_negatives))
print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")


# In[72]:


evaluate_approach_and_graph(df_over, file_tag="class_pos_covid_Balancing_A2")


# ## Approach 3
# 
# Oversampling by SMOTE

# In[70]:


target = "CovidPos"

df_positives: Series = df[df[target] == 1]
df_negatives: Series = df[df[target] == 0]
RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = df.pop(target).values
X: np.ndarray = df.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = pd.concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(df.columns) + [target]

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)


# In[73]:


evaluate_approach_and_graph(df_smote, file_tag="class_pos_covid_Balancing_A3")






from numpy import ndarray
from pandas import DataFrame, read_csv, concat
from matplotlib.pyplot import savefig, show, figure
from sklearn.model_selection import train_test_split
from dslabs_functions import dummify, get_variable_types, plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN, evaluate_approach



filename = "class_pos_covid.csv"
file_tag = "class_pos_covid"
data: DataFrame = read_csv(
    "datasets/class_pos_covid.csv", na_values="", parse_dates=True, dayfirst=True
)
vars: dict[str, list] = get_variable_types(data)

df: DataFrame = data.dropna(how="any", inplace=False)

binaries: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1, "Female":0, "Male":1}
encoding: dict[str, dict[str, int]] = {}
for bin_col in vars["binary"]:
	encoding[bin_col] = binaries

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

df: DataFrame = df.replace(encoding, inplace=False)

df = dummify(df, ["RaceEthnicityCategory"])

AgeCategory: dict[str, int] = {"Age 18 to 24":0, "Age 80 or older":12}
for i in range(11):
	AgeCategory[f'Age {i*5 + 25} to {i*5 + 29}'] = i+1
AgeCategory
encoding: dict[str, dict[str, int]] = {"AgeCategory":AgeCategory}

df: DataFrame = df.replace(encoding, inplace=False)

TetanusLast10Tdap: dict[str, int] = {
	"Yes, received tetanus shot but not sure what type":2, 
	"No, did not receive any tetanus shot in the past 10 years":0, 
	"Yes, received Tdap":7, 
	"Yes, received tetanus shot, but not Tdap":3
}
encoding: dict[str, dict[str, int]] = {"TetanusLast10Tdap":TetanusLast10Tdap}
df: DataFrame = df.replace(encoding, inplace=False)

HadDiabetes: dict[str, int] = {
	"No, pre-diabetes or borderline diabetes":1, 
	"Yes, but only during pregnancy (female)":3, 
	"Yes":2, 
	"No":0
}
encoding: dict[str, dict[str, int]] = {"HadDiabetes":HadDiabetes}
df: DataFrame = df.replace(encoding, inplace=False)

RemovedTeeth: dict[str, int] = {"None of them":0, "1 to 5":2, "6 or more, but not all":18, "All":32}
encoding: dict[str, dict[str, int]] = {"RemovedTeeth":RemovedTeeth}
df: DataFrame = df.replace(encoding, inplace=False)

target = "CovidPos"

x = df.drop(columns=[target])
y = df[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
df_train = concat([X_train, y_train], axis=1)
df_test = concat([X_test,y_test], axis=1)

figure()
eval: dict[str, list] = evaluate_approach(df_train, df_test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval.png")
show()
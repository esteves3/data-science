from dslabs_functions import plot_bar_chart
from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv, DataFrame


filename = "datasets/class_credit_score.csv"
file_tag = "credit_score"
data: DataFrame = read_csv(filename, na_values="", index_col="ID")

print(data.shape)


# SECOND


figure(figsize=(4, 2))
values: dict[str, int] = {
    "nr records": data.shape[0], "nr variables": data.shape[1]}

plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"images/{file_tag}_records_variables.png")
show()

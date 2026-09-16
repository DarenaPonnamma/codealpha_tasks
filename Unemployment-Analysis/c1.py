import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def show_slide(seconds=5):
    plt.show(block=False)
    plt.pause(seconds)
    plt.close()

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Unemployment in India.csv")

print("Original Column Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

# Remove extra spaces before/after column names
df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())


# ============================================================
# 3. RENAME COLUMNS
# ============================================================

df.rename(columns={
    "Region": "State",
    "Estimated Unemployment Rate (%)": "UnemploymentRate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "LabourParticipation"
}, inplace=True)


# ============================================================
# 4. CONVERT DATA TYPES
# ============================================================

# Convert Date
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

# Convert numerical columns to numbers
df["UnemploymentRate"] = pd.to_numeric(
    df["UnemploymentRate"],
    errors="coerce"
)

df["Employed"] = pd.to_numeric(
    df["Employed"],
    errors="coerce"
)

df["LabourParticipation"] = pd.to_numeric(
    df["LabourParticipation"],
    errors="coerce"
)


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

df.dropna(inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 6. BASIC INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 7. NATIONAL UNEMPLOYMENT TREND
# ============================================================

national_trend = (
    df.groupby("Date")["UnemploymentRate"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=national_trend,
    x="Date",
    y="UnemploymentRate",
    marker="o"
)

plt.axvline(
    pd.Timestamp("2020-03-01"),
    linestyle="--",
    label="COVID-19 Start"
)

plt.title("Average Unemployment Rate in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
show_slide(5)


# ============================================================
# 8. RURAL VS URBAN UNEMPLOYMENT
# ============================================================

area_trend = (
    df.groupby(["Date", "Area"])["UnemploymentRate"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=area_trend,
    x="Date",
    y="UnemploymentRate",
    hue="Area",
    marker="o"
)

plt.axvline(
    pd.Timestamp("2020-03-01"),
    linestyle="--",
    label="COVID-19 Start"
)

plt.title("Rural vs Urban Unemployment Trends")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
show_slide(5)


# ============================================================
# 9. COVID-19 PERIOD ANALYSIS
# ============================================================

covid_period = df[
    (df["Date"] >= pd.Timestamp("2020-03-01")) &
    (df["Date"] <= pd.Timestamp("2020-06-30"))
]

before_covid = df[
    df["Date"] < pd.Timestamp("2020-03-01")
]

covid_average = covid_period["UnemploymentRate"].mean()
before_average = before_covid["UnemploymentRate"].mean()

increase = covid_average - before_average

print("\n================ COVID-19 ANALYSIS ================")

print(
    "Average unemployment before COVID:",
    round(before_average, 2), "%"
)

print(
    "Average unemployment during COVID:",
    round(covid_average, 2), "%"
)

print(
    "Change in unemployment:",
    round(increase, 2), "%"
)


# ============================================================
# 10. STATE-WISE COVID-19 ANALYSIS
# ============================================================

state_covid = (
    covid_period
    .groupby("State")["UnemploymentRate"]
    .mean()
    .sort_values(ascending=False)
)

print("\nState-wise COVID-19 Unemployment:")
print(state_covid)


plt.figure(figsize=(10, 8))

sns.barplot(
    x=state_covid.values,
    y=state_covid.index
)

plt.title(
    "Average Unemployment Rate by State During COVID-19"
)

plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")

plt.tight_layout()
show_slide(5)


# ============================================================
# 11. TOP 10 STATES
# ============================================================

top_10 = state_covid.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_10.values,
    y=top_10.index
)

plt.title(
    "Top 10 States with Highest Unemployment During COVID-19"
)

plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")

plt.tight_layout()
show_slide(5)


# ============================================================
# 12. MONTHLY / SEASONAL PATTERN
# ============================================================

df["Month"] = df["Date"].dt.month

monthly_average = (
    df.groupby("Month")["UnemploymentRate"]
    .mean()
)

print("\nMonthly Average Unemployment:")
print(monthly_average)


plt.figure(figsize=(10, 5))

sns.barplot(
    x=monthly_average.index,
    y=monthly_average.values
)

plt.title("Seasonal Pattern of Unemployment Rate")

plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()
show_slide(5)


# ============================================================
# 13. AVERAGE UNEMPLOYMENT BY AREA
# ============================================================

area_average = (
    df.groupby("Area")["UnemploymentRate"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment by Area:")
print(area_average)


plt.figure(figsize=(7, 5))

sns.barplot(
    x=area_average.index,
    y=area_average.values
)

plt.title("Average Unemployment Rate: Rural vs Urban")

plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")

plt.tight_layout()
show_slide(5)


# ============================================================
# 14. LABOUR PARTICIPATION VS UNEMPLOYMENT
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="LabourParticipation",
    y="UnemploymentRate",
    hue="Area"
)

plt.title(
    "Labour Participation Rate vs Unemployment Rate"
)

plt.xlabel("Labour Participation Rate (%)")
plt.ylabel("Unemployment Rate (%)")

plt.tight_layout()
show_slide(5)


# ============================================================
# 15. CORRELATION ANALYSIS
# ============================================================

correlation = df[
    [
        "UnemploymentRate",
        "Employed",
        "LabourParticipation"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation)


plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Between Employment Indicators")

plt.tight_layout()
show_slide(5)


# ============================================================
# 16. HIGHEST UNEMPLOYMENT
# ============================================================

highest_index = df["UnemploymentRate"].idxmax()

print("\nHighest Unemployment Record:")
print(df.loc[highest_index])


# ============================================================
# 17. LOWEST UNEMPLOYMENT
# ============================================================

lowest_index = df["UnemploymentRate"].idxmin()

print("\nLowest Unemployment Record:")
print(df.loc[lowest_index])


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("       UNEMPLOYMENT ANALYSIS SUMMARY")
print("==========================================")

print(
    "Average Before COVID:",
    round(before_average, 2), "%"
)

print(
    "Average During COVID:",
    round(covid_average, 2), "%"
)

print(
    "Change:",
    round(increase, 2), "%"
)

print(
    "Highest COVID-affected State:",
    state_covid.idxmax()
)

print(
    "Highest Recorded Unemployment:",
    round(df["UnemploymentRate"].max(), 2), "%"
)

print(
    "Lowest Recorded Unemployment:",
    round(df["UnemploymentRate"].min(), 2), "%"
)

print(
    "Area with Higher Average Unemployment:",
    area_average.idxmax()
)

print("==========================================")
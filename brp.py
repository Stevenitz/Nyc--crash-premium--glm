# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:49:59 2026

@author: Stevenitzz
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats

Crash= pd.read_csv(r'C:\Users\Stevenitzz\OneDrive\Desktop\Motor_Vehicle_Collisions_-_Crashes.csv',low_memory=False ,parse_dates=['CRASH DATE'])

print(f"New data shape: {Crash.shape}")
print(f"Columns: {Crash.columns.tolist()}")

# Columns I found Interesting
keep_columns = [
    'CRASH DATE',
    'CRASH TIME', 
    'BOROUGH',
    'NUMBER OF PERSONS INJURED',
    'NUMBER OF PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED',
    'NUMBER OF PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED',
    'NUMBER OF CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED',
    'NUMBER OF MOTORIST KILLED',
    'CONTRIBUTING FACTOR VEHICLE 1',
    'VEHICLE TYPE CODE 1'
]

# Filter to just these columns
Crash = Crash[keep_columns].copy()
print(f"Cleaned shape: {Crash.shape}")






# Extract year from CRASH DATE
Crash["year"] = Crash["CRASH DATE"].dt.year


Crash["BOROUGH"] = Crash["BOROUGH"].fillna("Unknown")



Crash["VEHICLE TYPE CODE 1"].value_counts().head(10)
Crash["CONTRIBUTING FACTOR VEHICLE 1"].value_counts().head(10)


injury_cols = [
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED",
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED"
]
Crash[injury_cols] = Crash[injury_cols].fillna(0)


Crash['total_injuries'] = Crash['NUMBER OF PERSONS INJURED'] + Crash['NUMBER OF PERSONS KILLED']

Crash.info()
Crash.head()
Crash.describe()

Crash['date'] = pd.to_datetime(Crash['CRASH DATE'])
daily_counts = Crash.groupby(['BOROUGH', 'date']).size().reset_index(name='daily_crashes')

# STEP 2: Add predictors
daily_counts['year'] = daily_counts['date'].dt.year
daily_counts['month'] = daily_counts['date'].dt.month
daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
daily_counts['is_weekend'] = (daily_counts['day_of_week'] >= 5).astype(int)

# Use ALL complete years 
model_data = daily_counts[daily_counts['year'].between(2012, 2025)].copy()

# STEP 3: Frequency Model (Negative Binomial)
from statsmodels.discrete.discrete_model import NegativeBinomial



freq_model2 = NegativeBinomial.from_formula('daily_crashes ~ C(BOROUGH) ',data=model_data).fit()

model3=NegativeBinomial.from_formula('daily_crashes ~ C(BOROUGH) + C(year)',data=model_data).fit()



print("=== FREQUENCY MODELS (Negative Binomial) ===")

print(freq_model2.summary())
print(model3.summary())   # winner has double the explaing power



#residuals log scale funnel is fine
# Residual diagnostics

# Residuals vs Fitted



fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plot 1: Residuals vs Fitted
axes[0].scatter(model3.fittedvalues, model3.resid, alpha=0.3)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Plot 2: QQ Plot
stats.probplot(model3.resid, dist="norm", plot=axes[1])
axes[1].set_title('QQ Plot')

plt.tight_layout()
plt.show()



print("Min predicted:", model3.fittedvalues.min())
print("Max predicted:", model3.fittedvalues.max())
print("Actual daily crashes range:", model_data['daily_crashes'].min(), "to", model_data['daily_crashes'].max())





# Find the day with largest residual
max_resid_idx = model3.resid.idxmax()
outlier_day = model_data.iloc[max_resid_idx]
print("Outlier day:", outlier_day[['BOROUGH', 'date', 'daily_crashes']])
print("Predicted:", model3.fittedvalues.iloc[max_resid_idx])
print("Residual:", model3.resid.iloc[max_resid_idx])


# Add model selection metrics




from statsmodels.tools.eval_measures import aic, bic

# After fitting models, compare them systematically
model_names = ['Borough Only', 'Borough + Year']
models = [freq_model2, model3]

for name, model in zip(model_names, models):
    print(f"\n{name}:")
    print(f"AIC: {model.aic:.2f}, BIC: {model.bic:.2f}")
    print(f"LLR p-value: {model.llr_pvalue:.4f}")
    print(f"Log-Likelihood: {model.llf:.2f}")




#severity

# Filter to years 2012-2025 BEFORE aggregating
Crash_filtered = Crash[Crash['year'].between(2012, 2025)].copy()

#  create daily injuries from filtered data
daily_injuries = Crash_filtered.groupby(['BOROUGH', 'date'])['total_injuries'].sum().reset_index()
daily_injuries = daily_injuries.rename(columns={'total_injuries': 'daily_injuries'})

model_data = daily_counts.merge(daily_injuries, on=['BOROUGH', 'date'], how='left')
model_data['daily_injuries'] = model_data['daily_injuries'].fillna(0)

# Calculate daily severity (injuries per crash)
model_data['severity'] = model_data['daily_injuries'] / model_data['daily_crashes']
model_data['severity'] = model_data['severity'].replace([np.inf, -np.inf], np.nan)




#predictions

# Filter out 2026 from model_data before predicting
model_data_for_pred = model_data[model_data['year'].between(2012, 2025)].copy()

# Get predictions
model_data_for_pred['predicted_daily_crashes'] = model3.predict(model_data_for_pred)

# Calculate severity (average injuries per crash by borough)
severity_by_borough = model_data_for_pred[model_data_for_pred['severity'] > 0].groupby('BOROUGH')['severity'].mean()

# Add to data
model_data_for_pred['predicted_severity'] = model_data_for_pred['BOROUGH'].map(severity_by_borough)

# Calculate pure premium
model_data_for_pred['predicted_pure_premium'] = (
    model_data_for_pred['predicted_daily_crashes'] * 
    model_data_for_pred['predicted_severity']
)

#  results
print(model_data_for_pred[['BOROUGH', 'year', 'predicted_daily_crashes', 
                           'predicted_severity', 'predicted_pure_premium']].head())


yearly_results = model_data_for_pred.groupby(['BOROUGH', 'year']).agg(
    total_predicted_crashes=('predicted_daily_crashes', 'sum'),
    avg_severity=('predicted_severity', 'mean'),
    total_predicted_injuries=('predicted_pure_premium', 'sum')
).reset_index()

print(yearly_results.head(20))



# Example assumptions (simplified for illustration)
cost_per_minor_injury = 15000    # Medical + pain & suffering
cost_per_major_injury = 75000    # Serious injuries
cost_per_fatality = 500000       # Fatalities

# Since I don't have injury severity breakdown, use weighted average
# data pattern: mostly minor, some serious, few fatalities
avg_cost_per_injury = 35000  # Reasonable placeholder:
    
    
yearly_results['expected_dollar_losses'] = yearly_results['total_predicted_injuries'] * avg_cost_per_injury

# Add exposure if you had it (number of insured cars)
# yearly_results['pure_premium_per_car'] = yearly_results['expected_dollar_losses'] / number_of_cars

print(yearly_results[['BOROUGH', 'year', 'total_predicted_injuries', 
                      'expected_dollar_losses']].head())



# Pivot table to compare boroughs side by side
pivot_table = yearly_results.pivot_table(
    index='BOROUGH',
    columns='year',
    values='total_predicted_injuries',
    aggfunc='sum'
)
print(pivot_table)

# for specific year:
    # Show all boroughs for a specific year (e.g., 2023)
year_2023 = yearly_results[yearly_results['year'] == 2023]
print(year_2023[['BOROUGH', 'total_predicted_crashes', 'avg_severity', 'total_predicted_injuries']])
    
#add covid years and 2025

#HEATMAP/VISUALS for port mostly 

#1. premiums by borough and year- more honest

# Pivot your yearly_results for heatmap
premium_matrix = yearly_results.pivot(
    index='BOROUGH',
    columns='year', 
    values='total_predicted_injuries'  # or 'expected_dollar_losses'
)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(premium_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
            linewidths=0.5, cbar_kws={'label': 'Expected Injuries'})
plt.title('Expected Injuries by Borough and Year')
plt.xlabel('Year')
plt.ylabel('Borough')
plt.tight_layout()
plt.show()


#by $ amounts ezier to interpet cause no assumption andis my model and shows actual work
# Add dollar conversion first
yearly_results['expected_dollar_losses'] = yearly_results['total_predicted_injuries'] * 35000

dollar_matrix = yearly_results.pivot(
    index='BOROUGH',
    columns='year',
    values='expected_dollar_losses'
)

plt.figure(figsize=(12, 8))
sns.heatmap(dollar_matrix/1e6, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=0.5, cbar_kws={'label': 'Million $'})
plt.title('Expected Losses by Borough and Year (Millions $)')
plt.xlabel('Year')
plt.ylabel('Borough')
plt.tight_layout()
plt.show()












import pandas as pd
from scipy.stats import chisquare
df = pd.read_csv('laughter-corpus.csv')
from math import sqrt

# --- Chi Square Laugh Time Gender ---
print("--- Q1 Chi Square Number of Laughs Gender --- ")
laugh_count_gender = df['Gender'].value_counts()
total_laughs = len(df['Gender'])
male_ratio, female_ratio = 57/120, 63/120

observed_gender_counts = [laugh_count_gender['Male'], laugh_count_gender['Female']]
expected_gender_counts = [total_laughs * male_ratio, total_laughs * female_ratio]

chi_gender = chisquare(observed_gender_counts, f_exp=expected_gender_counts, ddof=1)
print(chi_gender)
# --- Chi Square Laugh Time Caller Role ---
print("\n--- Q2 Chi Square Number of Laughs Call Role ---")
laugh_count_role = df['Role'].value_counts()

observed_role_counts = [laugh_count_role['Caller'], laugh_count_role['Receiver']]
expected_role_counts = [0.5*total_laughs, 0.5*total_laughs]
chi_role = chisquare(observed_role_counts, f_exp=expected_role_counts, ddof=1)
print(chi_role)
# --- Student T Laugh Time Gender ---
print("\n--- Q3 Student T Laugh Time Gender ---")
mean_time_gender = (df.groupby('Gender')['Duration']).mean()
std_time_gender = (df.groupby('Gender')['Duration']).std()

t_gender_numerator =  mean_time_gender['Female'] - mean_time_gender['Male']
t_gender_denominator = sqrt((std_time_gender['Female']**2 / laugh_count_gender['Female'])
                            + (std_time_gender['Male']**2 / laugh_count_gender['Male']))
t_gender = t_gender_numerator / t_gender_denominator
print(t_gender)

# --- Student T Laugh Time Caller Role ---
print("\n--- Q4 Student T Laugh Time Call Role ---")
mean_time_role = (df.groupby('Role')['Duration']).mean()
std_time_role = (df.groupby('Role')['Duration']).std()

t_role_numerator = mean_time_role['Caller'] - mean_time_role['Receiver']
t_role_denominator = sqrt((std_time_role['Caller']**2 / laugh_count_role['Caller'])
                          + (std_time_role['Receiver']**2) / laugh_count_role['Receiver'])
t_role = t_role_numerator / t_role_denominator
print(t_role)



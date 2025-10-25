import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./bank-full.csv')

def graph_age_distribution(column):
    plt.figure(figsize=(8, 6))
    plt.hist(column, bins=20, color='blue', edgecolor='black')
    plt.title('Age Distribution of Bank Customers')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.show()

def graph_job_distribution(column):
    plt.figure(figsize=(8, 6))
    job_counts = column.value_counts()
    job_counts.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Job Distribution of Bank Customers')
    plt.xlabel('Job Type')
    plt.ylabel('Number of Customers')
    plt.show()

def graph_marital_status(column):
    plt.figure(figsize=(8, 6))
    marital_counts = column.value_counts()
    marital_counts.plot(kind='bar', color='green', edgecolor='black')
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Number of Customers')
    plt.show()

def graph_balance_distribution(column):
    plt.figure(figsize=(8, 6))
    plt.hist(column, bins=30, color='purple', edgecolor='black')
    plt.title('Balance Distribution of Bank Customers')
    plt.xlabel('Balance')
    plt.ylabel('Number of Customers')
    plt.show()

def boxplot(column):
    plt.figure(figsize=(8, 6))
    plt.boxplot(column, vert=True, patch_artist=True)
    plt.title(f"{column.name} distribution boxplot")
    plt.ylabel('Values')
    plt.show()

# boxplot(df['age'])
# boxplot(df['balance'])

# graph_age_distribution(df['age'])
# graph_job_distribution(df['job'])
# graph_marital_status(df['marital'])
# graph_balance_distribution(df['balance'])

# boxplot(df['duration'])

# num_df = df.select_dtypes(include=['int64', 'float64'])

# stats_summary = pd.DataFrame({
#     'Minimum': num_df.min(),
#     'Maximum': num_df.max(),
#     'Mean': num_df.mean(),
#     'Median': num_df.median(),
#     'Mode': num_df.mode().iloc[0],
#     'Standard Deviation': num_df.std()
# })

# print("=== Measures of Central Tendency ===")
# print(stats_summary)
# print("\n")

# boxplot(df['previous'])

# graph_balance_distribution(df['education'])

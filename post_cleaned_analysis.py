import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./cleaned_bank_data.csv')

def boxplot(column):
    plt.figure(figsize=(8, 6))
    plt.boxplot(column, vert=True, patch_artist=True)
    plt.title(f"{column.name} distribution boxplot")
    plt.ylabel('Values')
    plt.show()

def distribution_histogram(column, bins=30, color='blue', title='Distribution Histogram', xlabel='Values', ylabel='Frequency'):
    plt.figure(figsize=(8, 6))
    plt.hist(column, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

boxplot(df['age'])
boxplot(df['balance'])

distribution_histogram(df['age'], bins=20, color='blue', title='Age Distribution of Bank Customers', xlabel='Age', ylabel='Number of Customers')
distribution_histogram(df['job'], bins=len(df['job'].unique()), color='orange', title='Job Distribution of Bank Customers', xlabel='Job Type', ylabel='Number of Customers')

distribution_histogram(df['marital'], bins=len(df['marital'].unique()), color='green', title='Marital Status Distribution', xlabel='Marital Status', ylabel='Number of Customers')

num_df = df.select_dtypes(include=['int64', 'float64'])

stats_summary = pd.DataFrame({
    'Minimum': num_df.min(),
    'Maximum': num_df.max(),
    'Mean': num_df.mean(),
    'Median': num_df.median(),
    'Mode': num_df.mode().iloc[0],
    'Standard Deviation': num_df.std()
})

print("=== Measures of Central Tendency ===")
print(stats_summary)
print("\n")

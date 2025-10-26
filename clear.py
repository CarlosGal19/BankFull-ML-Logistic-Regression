import pandas as pd

df = pd.read_csv('./bank-full.csv')

df = df[df['balance'] < 80000]
df = df[df['duration'] < 3500]
df = df[df['previous'] < 100]

df = pd.get_dummies(df, columns=['job', 'marital', 'contact', 'month', 'poutcome'], drop_first=True)

df['default'] = df['default'].map({'yes': 1, 'no': 0})
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
df['Target'] = df['Target'].map({'yes': 1, 'no': 0})

education_order = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}

df['education'] = df['education'].map(education_order)

target_1_rows = df[df['Target'] == 1]
df = pd.concat([df, target_1_rows, target_1_rows], ignore_index=True)
df = pd.concat([df, target_1_rows, target_1_rows], ignore_index=True)

df.to_csv('./cleaned_bank_data.csv', index=False)

import pandas as pd
from dateutil.relativedelta import relativedelta

# Load your dataset
file_path = "C:\\Users\\prana\\Downloads\\telephn_churn\\WA_Fn-UseC_-Telco-Customer-Churn (2).csv"
df = pd.read_csv(file_path)

# Define base date
base_date = pd.to_datetime("2015-01-01")

# Create JoinDate = base_date + (tenure-1 months)
df['JoinDate'] = [base_date + relativedelta(months=int(x)-1) for x in df['tenure']]

# Extract useful date parts
df['JoinYear'] = df['JoinDate'].dt.year
df['JoinMonth'] = df['JoinDate'].dt.month_name()

# Preview
print(df[['tenure', 'JoinDate', 'JoinYear', 'JoinMonth']].head(10))


# Check sample output
print(df[['tenure', 'JoinDate', 'JoinYear', 'JoinMonth']].sample(5))
df.to_csv('check2.csv',index=False)
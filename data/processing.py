#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime, timedelta

# Define the date range
start_date = datetime(2006, 9, 4)
end_date = datetime(2025, 1, 3)

# Generate all dates in the range
all_dates = pd.date_range(start=start_date, end=end_date)

# Filter out Saturdays and Sundays
weekdays = all_dates[~all_dates.weekday.isin([5, 6])]

# Create a DataFrame and write it to a CSV file
df = pd.DataFrame(weekdays, columns=["Data"])
csv_path = "01_weekdays_df.csv"
df.to_csv(csv_path, index=False)

csv_path


# In[2]:


weekdays_df= pd.read_csv('01_weekdays_df.csv')
bshi_data = pd.read_csv('BHSI.csv')

# Ensure both datasets have a proper datetime format for merging
weekdays_df["Data"] = pd.to_datetime(weekdays_df["Data"])
bshi_data["Date"] = pd.to_datetime(bshi_data["Date"])

# Merge the two datasets based on the date, with BHSI values or "absent" for missing dates
merged_df = weekdays_df.merge(bshi_data, how="left", left_on="Data", right_on="Date")

# Fill missing BHSI values with "absent" and drop the extra 'Date' column from the BHSI file
merged_df["BHSI"] = merged_df["BHSI"].fillna("")
merged_df = merged_df.drop(columns=["Date"])

# Save the merged result to a new CSV file
final_csv_path = "02_merged_weekdays_bhsi.csv"
merged_df.to_csv(final_csv_path, index=False)


# In[3]:


import pandas as pd

# Load the dataset
df = pd.read_csv('02_merged_weekdays_bhsi.csv')

# Preprocess the dataset
df['Data'] = pd.to_datetime(df['Data'])
df['BHSI'] = pd.to_numeric(df['BHSI'].str.replace(',', ''), errors='coerce')
df['Weekday'] = df['Data'].dt.weekday  # Monday=0, Sunday=6

# Define the imputation function with all rules
def impute_weekdays_with_rules(group):
    weekdays = group[group['Weekday'] < 5]  # Filter weekdays (Monday to Friday)
    weekends = group[group['Weekday'] >= 5]  # Retain weekends untouched

    if weekdays.empty:
        return group  # Return unchanged if no weekdays exist

    # Extract weekday values
    monday = weekdays.loc[weekdays['Weekday'] == 0, 'BHSI']
    tuesday = weekdays.loc[weekdays['Weekday'] == 1, 'BHSI']
    wednesday = weekdays.loc[weekdays['Weekday'] == 2, 'BHSI']
    thursday = weekdays.loc[weekdays['Weekday'] == 3, 'BHSI']
    friday = weekdays.loc[weekdays['Weekday'] == 4, 'BHSI']

    # Rule 2: Extrapolate Monday and Friday
    if monday.isnull().any() and tuesday.notnull().any() and wednesday.notnull().any():
        weekdays.loc[weekdays['Weekday'] == 0, 'BHSI'] = 2 * tuesday.values[0] - wednesday.values[0]
    if friday.isnull().any() and thursday.notnull().any() and wednesday.notnull().any():
        weekdays.loc[weekdays['Weekday'] == 4, 'BHSI'] = 2 * thursday.values[0] - wednesday.values[0]

    # Rule 3: If all weekdays are missing, fill them with 0
    if weekdays['BHSI'].isnull().all():
        weekdays['BHSI'].fillna(0, inplace=True)

    # Rule 4: If only one value exists, propagate it across weekdays
    elif weekdays['BHSI'].count() == 1:
        weekdays['BHSI'].fillna(weekdays['BHSI'].dropna().iloc[0], inplace=True)

    # Rule 5: Handle two missing values
    elif weekdays['BHSI'].isnull().sum() == 2:
        # Handle missing Monday and Tuesday
        if monday.isnull().any() and tuesday.isnull().any() and wednesday.notnull().any() and thursday.notnull().any() and friday.notnull().any():
            value = (wednesday.values[0] + thursday.values[0] + friday.values[0]) / 3
            weekdays.loc[weekdays['Weekday'] == 0, 'BHSI'] = value  # Monday
            weekdays.loc[weekdays['Weekday'] == 1, 'BHSI'] = value  # Tuesday
        
        # Handle missing Thursday and Friday
        if thursday.isnull().any() and friday.isnull().any() and monday.notnull().any() and tuesday.notnull().any() and wednesday.notnull().any():
            value = (monday.values[0] + tuesday.values[0] + wednesday.values[0]) / 3
            weekdays.loc[weekdays['Weekday'] == 3, 'BHSI'] = value  # Thursday
            weekdays.loc[weekdays['Weekday'] == 4, 'BHSI'] = value  # Friday

    # Rule 6: Handle three weekday values
    elif weekdays['BHSI'].notnull().sum() == 3:
        avg_value = weekdays['BHSI'].mean()
        weekdays['BHSI'].fillna(avg_value, inplace=True)

    # Rule 7: Handle two weekday values
    elif weekdays['BHSI'].notnull().sum() == 2:
        avg_value = weekdays['BHSI'].mean()
        weekdays['BHSI'].fillna(avg_value, inplace=True)

    # Combine weekdays and weekends
    return pd.concat([weekdays, weekends]).sort_index()

# Apply the imputation logic weekly
df = df.groupby(pd.Grouper(key='Data', freq='W')).apply(impute_weekdays_with_rules).reset_index(drop=True)

# Save the imputed dataset to a CSV file
output_file = '03_imputed_weekdays_bhsi.csv'
df.to_csv(output_file, index=False)
print(f"Imputed dataset saved to {output_file}")



# In[4]:


import pandas as pd

# Load the uploaded CSV file
file_path = "03_imputed_weekdays_bhsi.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
df.head()

# Identify missing values in the "BHSI" column
missing_indices = df[df['BHSI'].isna()].index

# Filter missing values occurring on Tuesday (1), Wednesday (2), or Thursday (3)
missing_weekdays = df.loc[missing_indices, :].query("Weekday in [1, 2, 3]")

# Verify if imputation was correctly applied
imputation_checks = []

for idx in missing_weekdays.index:
    if idx > 0 and idx < len(df) - 1:
        prev_value = df.loc[idx - 1, 'BHSI']
        next_value = df.loc[idx + 1, 'BHSI']
        expected_value = (prev_value + next_value) / 2
        actual_value = df.loc[idx, 'BHSI']

        imputation_checks.append({
            "Date": df.loc[idx, "Data"],
            "Weekday": df.loc[idx, "Weekday"],
            "Expected Imputed Value": expected_value,
            "Actual Value": actual_value,
            "Correctly Imputed": actual_value == expected_value
        })


# In[7]:


# Perform the correct imputation for missing values on Tuesday, Wednesday, or Thursday
for idx in imputation_results_df.index:
    actual_idx = df[df["Data"] == imputation_results_df.loc[idx, "Date"]].index[0]
    if actual_idx > 0 and actual_idx < len(df) - 1:
        prev_value = df.loc[actual_idx - 1, 'BHSI']
        next_value = df.loc[actual_idx + 1, 'BHSI']
        df.loc[actual_idx, 'BHSI'] = (prev_value + next_value) / 2

# Save the corrected file
corrected_file_path = "03_corrected_imputed_weekdays_bhsi.csv"
df.to_csv(corrected_file_path, index=False)

# Provide the download link
corrected_file_path


# In[6]:


import pandas as pd
import numpy as np

headers = ['Data', 'BHSI']

df = pd.read_csv('03_corrected_imputed_weekdays_bhsi.csv',
#                 skiprows=6, skipfooter=9,
                 engine='python')



# In[ ]:


for i in range(957):
    if i == 0:
        df1=df[5*i:5*(i+1)]
        df2=pd.concat([df1["BHSI"]])#,df1["cape_5TC_CCURMON"],df1["cape_5TC_CCURQ"]])
    else:
        df3=df[5*i:5*(i+1)]
        df3=pd.concat([df3["BHSI"]])#,df3["cape_5TC_CCURMON"],df3["cape_5TC_CCURQ"]])
        df2=np.vstack([df2,df3])


# In[ ]:


arr = np.array(df2)
df2 = pd.DataFrame(arr)
df2.to_csv('04_input_BHSI.csv', index=False)


# In[ ]:


# Ensure numeric values
df = pd.read_csv('04_input_BHSI.csv')

df = df.apply(pd.to_numeric, errors='coerce')

# Create a new column for comparison results
df['labelling'] = 0  # Initialize with 0

# Compare row averages
for i in range(len(df) - 1):
    current_avg = df.iloc[i, :5].mean()
    next_avg = df.iloc[i + 1, :5].mean()
    df.at[i, 'labelling'] = 0 if current_avg > next_avg else 1

# Save the resulting dataset
df.to_csv('05_comparison_results.csv', index=False)

# Display first few rows
print(df.head())


# In[ ]:


import pandas as pd
import numpy as np

headers = ['Date', 'BHSI']

dff = pd.read_csv('02_merged_weekdays_bhsi.csv',
#                 skiprows=6, skipfooter=9,
                 engine='python')
print(dff.head())


# In[ ]:


for i in range(957):
    if i == 0:
        dff1=dff[5*i:5*(i+1)]
        dff2=pd.concat([dff1["BHSI"]])#,df1["cape_5TC_CCURMON"],df1["cape_5TC_CCURQ"]])
    else:
        dff3=dff[5*i:5*(i+1)]
        dff3=pd.concat([dff3["BHSI"]])#,df3["cape_5TC_CCURMON"],df3["cape_5TC_CCURQ"]])
        dff2=np.vstack([dff2,dff3])


# In[ ]:


arrr = np.array(dff2)
dff2 = pd.DataFrame(arrr)
dff2.to_csv('06_input_original_BHSI.csv', index=False)


# In[ ]:


import pandas as pd

# Load the uploaded file
file_path = '06_input_original_BHSI.csv'
data = pd.read_csv(file_path)

# Replace commas with empty strings and convert to numeric, coercing errors to handle non-numeric entries as NaN
data_cleaned = data.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Count missing values for each row
missing_counts = data_cleaned.isnull().sum(axis=1)

# Initialize a list to store row numbers and their reasons
rows_to_include = []
row_numbers = []

# Apply the conditions
for idx, count in missing_counts.items():
    if 2 <= count <= 4:
        rows_to_include.append(idx)
    elif count == 5:
        rows_to_include.append(idx)
        if idx > 0:  # Ensure not to access an invalid index
            rows_to_include.append(idx - 1)

# Remove duplicates and sort row indices
rows_to_include = sorted(set(rows_to_include))

# Create a new column to store the row numbers
data_cleaned["Row Number"] = range(1, len(data_cleaned) + 1)
data_cleaned["Selection Flag"] = data_cleaned.index.isin(rows_to_include).astype(int)

# Filter the rows that satisfy the conditions
filtered_data = data_cleaned[data_cleaned["Selection Flag"] == 1]

# Save the filtered data with the added column
filtered_data.to_csv('07_filtered_input_original_BHSI.csv', index=False)


# In[ ]:


import pandas as pd

# Load the original and filtered data files
original_file_path = '05_comparison_results.csv'
filtered_file_path = '07_filtered_input_original_BHSI.csv'

# Load the datasets
original_data = pd.read_csv(original_file_path)
filtered_data = pd.read_csv(filtered_file_path)

# Extract the row numbers to delete from the 'Row Number' column in the filtered file
rows_to_delete = filtered_data['Row Number'].dropna().astype(int).tolist()

# Adjust the row numbers to 0-based index and remove them from the original file
removed_rows = original_data.iloc[[i - 1 for i in rows_to_delete], :]
remaining_data = original_data.drop(index=[i - 1 for i in rows_to_delete], errors='ignore')

# Save the edited data and the removed rows to separate files
edited_file_path = '08_edited_file_comparison_results.csv'
removed_rows_file_path = '09_removed_rows.csv'

remaining_data.to_csv(edited_file_path, index=False)
removed_rows.to_csv(removed_rows_file_path, index=False)

edited_file_path, removed_rows_file_path


# In[ ]:


import pandas as pd

# Load the CSV file
file_path = '08_edited_file_comparison_results.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Reorder columns: move the last column to the first position
columns = df.columns.tolist()
columns = [columns[-1]] + columns[:-1]  # Reorder columns
df_reordered = df[columns]

# Save the modified dataframe to a new file
output_file_path = '10_input_BHSI.csv'  # Replace with your desired output path
df_reordered.to_csv(output_file_path, index=False)

print(f"Reordered file saved to: {output_file_path}")


# In[ ]:





# In[ ]:





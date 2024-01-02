#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Importing the dataset and to avoid unicode error we are encoding with 'latin-1'

# In[2]:


df=pd.read_csv(r"C:\Users\Asus\Downloads\ISRO mission launches.csv", encoding='latin-1')
df.head()


# The second and the third column has an error, hence it has been swapped  

# In[3]:


df.rename(columns={'Launch Vehicle':'Launch Date', 'Launch Date':'Launch Vehicle'}, inplace=True)
df.head()


# Identifying the shape of the dataframe

# In[4]:


df.shape


# Checking out all the column names

# In[5]:


df.columns


# Some basic information to get going

# In[6]:


df.info()


# Describes the description of the dataframe such as mean, std, freq, etc.

# In[7]:


df.describe(include='all')


# Summing up all the null values for each columns in the dataframe

# In[8]:


df.isnull().sum()


# In[9]:


[features for features in df.columns if df[features].isnull().sum()>0]


# Heatmap for Nullvalues

# In[10]:


sns.heatmap(df.isnull(),cbar=False, cmap='plasma')


# In[11]:


df.dtypes


# This snippet iterates through each column in the DataFrame that has an 'object' data type and then it prints a message indicating the column name, and then prints the occurence of each unique value in that column. 
# This is to explore categorical or textual data in the DataFrame and get a sense of the distribution of values in each categorical column

# In[12]:


value_counts_dict = {column: df[column].value_counts() for column in df.select_dtypes(include=['object']).columns}

for column, value_counts in value_counts_dict.items():
    print(f"\nValue counts for column: {column}")
    print(value_counts)


# # Data Preprocessing

# Filling up null values

# In[13]:


df['Orbit Type'].fillna('Unknown', inplace=True)
df['Application'].fillna('Unknown', inplace=True)


# In[14]:


df['Launch Date']=pd.to_datetime(df['Launch Date'])
df['Year']=df['Launch Date'].dt.year
df['Month']=df['Launch Date'].dt.month
df['Day']=df['Launch Date'].dt.day

print(df[['Year', 'Month', 'Day']].describe())


# # Exploratory Data Analysis

# The bottom snippet shows the most widely used applications

# In[15]:


# Seaborn Count Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Application', palette='Set3')  # Set3 is an example palette with unique colors
plt.xticks(rotation=90)
plt.title('Application popularity')
plt.show()

# Plotly Pie Chart
application_counts = df['Application'].value_counts()

# Define custom colors for each category
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']

fig = go.Figure(data=[go.Pie(labels=application_counts.index, values=application_counts.values, marker=dict(colors=colors))])
fig.update_layout(title='Applications')
fig.show()


# This snippet checks which Orbit type launches were successfull

# In[16]:


palette = 'Set3'

plt.figure(figsize=(12, 6))
ax = sns.histplot(data=df, x='Remarks', y='Orbit Type', hue='Orbit Type', multiple='stack', palette=palette)

# Adjust the legend position
ax.legend(bbox_to_anchor=(1.15, 1.0), title='Orbit Type')

plt.title('Histogram of Remarks vs Orbit Type')
plt.xticks(rotation=90)
plt.show()


# No. of launches per year

# In[17]:


plt.figure(figsize=(15, 5))
sns.lineplot(x=df['Year'].value_counts().sort_index().index, y=df['Year'].value_counts().sort_index().values)
plt.title("Number of Launches per Year (Line Chart)")
plt.xlabel("Year")
plt.ylabel("Number of Launches")
plt.xticks(rotation=90)

# Increase the tick frequency on the x-axis
plt.xticks(df['Year'].value_counts().sort_index().index, rotation=90)
plt.show()


# The code generates visualizations illustrating the distribution of launches across orbit types. 
# The first bar chart (`fig8`) provides a straightforward count of launches for each orbit type. 
# The second combined chart (`fig_combined`) goes further by presenting success rates alongside counts. 
# This approach offers a concise yet insightful view, allowing for a quick comparison of both the frequency 
# and success rates of launches across various orbit types.

# In[18]:


orbit_type_counts = df['Orbit Type'].value_counts()
fig8 = px.bar(x=orbit_type_counts.index, y=orbit_type_counts.values, 
              title='Orbit Type Counts')
fig8.show()

# Calculate success rates for each orbit type
orbit_success_rates = df.groupby('Orbit Type')['Remarks'].apply(lambda x: (x == 'Launch successful').mean()).reset_index()

# Combine counts and success rates in a single chart
fig_combined = px.bar(orbit_success_rates, x='Orbit Type', y='Remarks', color='Remarks',
                      title='Orbit Type Counts and Success Rates',
                      labels={'Remarks': 'Success Rate'},
                      height=400)

# Add counts as a line chart on the same figure
fig_combined.add_trace(px.line(x=orbit_type_counts.index, y=orbit_type_counts.values, 
                              labels={'y': 'Count'}, line_shape='linear').data[0])

# Update layout for better visualization
fig_combined.update_layout(yaxis=dict(tickformat="%", title='Success Rate'),
                           yaxis2=dict(title='Count', overlaying='y', side='right'),
                           legend=dict(title='Remarks'))

fig_combined.show()


# In[ ]:





# In[ ]:





# In[ ]:





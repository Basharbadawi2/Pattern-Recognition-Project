#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


df = pd.read_csv("C:/Users/basha/OneDrive/Desktop/Route Ai Eng Basma Reda/Classification & Regression Project/regression_project_data.csv")


# In[9]:


df


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


df.nunique()


# In[16]:


df.info()


# In[17]:


df.columns.tolist()


# # **Data Visualization**

# In[18]:


x = df['age']
import warnings
warnings.filterwarnings('ignore')


# In[19]:


sns.distplot(x)


# In[20]:


plt.figure(figsize=(40,8))
plt.title('Nationality Count Plot')
sns.countplot(x='nationality', data=df)
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.show()


# In[21]:


x = df['overall_rating']


# In[22]:


# Create the distribution plot
sns.distplot(x)
plt.tilte('overall_rating')


# In[23]:


df.hist(bins=50,figsize=(18,10))


# In[24]:


# Create the scatter plot
ax = sns.scatterplot(x='potential', y='value_euro', data=df, hue='potential', size='value_euro', palette='cool', sizes=(20, 200), legend=False)

# Add a regression line
sns.regplot(x='potential', y='value_euro', data=df, scatter=False, ax=ax, color='blue')

# Set titles and labels
plt.title('Potential vs. Value Scatter Plot')
plt.xlabel('Potential')
plt.ylabel('Value')


# # **Data preprocessing**

# In[25]:


columns_to_drop = [
    'full_name', 'birth_date', 'nationality', 'value_euro', 'wage_euro', 
    'preferred_foot', 'release_clause_euro', 'national_team', 'national_rating', 
    'national_team_position', 'national_jersey_number'
]


# In[26]:


df2 = df.drop(columns = columns_to_drop)


# In[27]:


# Dataframe after dropping columns
df2


# In[28]:


df2.isnull().sum()


# In[29]:


df2.info()


# In[ ]:





# In[ ]:





# In[80]:


df2['body_type'].value_counts()


# # **Label Encoding**

# In[30]:


# Apply Label Encoding to categorical columns
label_encoder = LabelEncoder()


# In[31]:


df2['name_encoder'] = label_encoder.fit_transform(df2['name'])


# In[32]:


df2['positions_encoder'] = label_encoder.fit_transform(df2['positions'])


# In[33]:


df2['body_type_encoder'] = label_encoder.fit_transform(df2['body_type'])


# In[34]:


# Apply LabelEncoder to each categorical column
for col in df2.select_dtypes(include=['object', 'category']).columns:
    df2[col] = label_encoder.fit_transform(df2[col])


# In[35]:


print(df2)


# In[36]:


# Specify the target coulmn
target_column = 'overall_rating'


# In[37]:


# Seprate features and target
X = df2.drop(columns=[target_column]) # Feature
Y = df2[target_column]


# In[38]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# # **Regression Models**

# In[39]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# In[40]:


# Initialize and train each regression model
models = {
    'Linear Regression': LinearRegression(),
    'KNN Regression': KNeighborsRegressor(),
    'SVR': SVR(),
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(base_estimator=DecisionTreeRegressor()),
}


# In[41]:


# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} Mean Squared Error: {mse:.2f}")


# In[ ]:




